import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from config import SERVER_HOST, SERVER_PORT, OLLAMA_URL, LLM_MODEL
from core.logging_config import configure_logging
from pipeline.retrieval import query_meeting
from pipeline.embeddings import _vectorstore_cache, _CACHE_MAX_SIZE
from eval.runner import setup_transcript, evaluate_question

configure_logging()


async def _check_ollama():
    """Verify Ollama is reachable and the configured model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if LLM_MODEL not in available:
                logging.warning(
                    f"Ollama is running but model '{LLM_MODEL}' was not found. "
                    f"Available models: {available}. "
                    f"Run: ollama pull {LLM_MODEL}"
                )
            else:
                logging.info(f"Ollama health check passed. Model '{LLM_MODEL}' is available.")
    except httpx.ConnectError:
        logging.error(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running before sending queries."
        )
    except Exception as e:
        logging.error(f"Ollama health check failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _check_ollama()
    yield


app = FastAPI(title="Meeting Copilot", version="2.0", lifespan=lifespan)

# One query runs at a time to avoid GPU memory contention
_gpu_semaphore = asyncio.Semaphore(1)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            meeting_text = data.get("meeting_text", "").strip()
            query = data.get("query", "").strip()
            logging.info(f"Received query: '{query[:80]}' | transcript_len={len(meeting_text)}")

            if not meeting_text or not query:
                await websocket.send_text("Error: Missing meeting_text or query.")
                continue

            if _gpu_semaphore.locked():
                await websocket.send_text(
                    "\n\n-> Another query is being processed. Please wait a moment and try again."
                )
                continue

            async with _gpu_semaphore:
                try:
                    async with asyncio.timeout(120):
                        await query_meeting(meeting_text, query, websocket)
                except asyncio.TimeoutError:
                    logging.error("Query timed out after 120 seconds.")
                    await websocket.send_text("\n\nError: Query timed out. Please try again.")
                except asyncio.CancelledError:
                    logging.info("Query cancelled (client disconnected mid-stream).")
                    raise
                except Exception as e:
                    logging.error(f"Query processing error: {e}")
                    await websocket.send_text(f"\n\nError: {str(e)}")

    except WebSocketDisconnect as e:
        logging.info(f"Client disconnected (code={e.code}, reason={e.reason!r})")
    except asyncio.CancelledError:
        logging.info("WebSocket task cancelled.")
        raise
    except Exception as e:
        logging.error(f"Unhandled WebSocket error: {type(e).__name__}: {e}")


@app.websocket("/ws/eval")
async def eval_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        transcript = data.get("transcript", "").strip()
        questions = [q.strip() for q in data.get("questions", []) if q.strip()]

        if not transcript or not questions:
            await websocket.send_json({"type": "error", "message": "Missing transcript or questions."})
            return

        ctx = await setup_transcript(transcript)

        for i, query in enumerate(questions):
            await websocket.send_json({"type": "progress", "index": i, "total": len(questions), "question": query})
            try:
                result = await evaluate_question(ctx, query)
                await websocket.send_json({"type": "result", "index": i, "total": len(questions), **result})
            except Exception as e:
                await websocket.send_json({"type": "result", "index": i, "total": len(questions), "question": query, "error": str(e)})

        await websocket.send_json({"type": "complete"})

    except WebSocketDisconnect:
        logging.info("Eval client disconnected.")
    except Exception as e:
        logging.error(f"Eval WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


@app.get("/debug/cache")
async def debug_cache():
    entries = []
    for key, (chunks, start_ts, index) in _vectorstore_cache.items():
        size_mb = (index.ntotal * index.d * 4) / (1024 ** 2)
        entries.append({
            "hash": key,
            "chunks": len(chunks),
            "vectors": index.ntotal,
            "size_mb": round(size_mb, 2),
        })
    total_mb = sum(e["size_mb"] for e in entries)
    return JSONResponse({"total_entries": len(entries), "max_entries": _CACHE_MAX_SIZE, "total_mb": round(total_mb, 2), "entries": entries})


app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
