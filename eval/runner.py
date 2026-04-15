"""
Per-question evaluation runner used by the /ws/eval WebSocket endpoint.

Usage pattern:
    ctx = await setup_transcript(transcript)          # once per transcript
    result = await evaluate_question(ctx, query)      # once per question
"""

import asyncio
import re
from dataclasses import dataclass
from functools import partial
from typing import Any

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    _LLMContextPrecisionWithoutReference,
    _ContextRelevance,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from config import OLLAMA_URL, LLM_MODEL, EMBEDDING_MODEL
from core.models import nlp
from pipeline.embeddings import get_or_build_vectorstore
from pipeline.retrieval import search_chunks_with_temporal_and_speaker
from pipeline.prompt import build_prompt
from utils.speaker import (
    build_speaker_registry,
    build_speaker_index,
    extract_speakers_from_text,
    remove_matched_speakers,
)

_SYNTHESIS_PHRASES = {
    "action item", "action items", "open question", "open questions",
    "next step", "next steps", "follow up", "follow-up", "key decision",
    "key decisions", "decision", "decisions", "summary", "summarize",
    "main point", "main points", "what was decided", "what did we decide",
    "todo", "to do", "to-do", "everything", "all", "entire", "whole",
}


class _MockWebSocket:
    async def send_text(self, _: str):
        pass


def _detect_synthesis(query: str) -> bool:
    q = query.lower()
    return any(re.search(r"\b" + re.escape(p) + r"\b", q) for p in _SYNTHESIS_PHRASES)


@dataclass
class TranscriptContext:
    """Pre-built, per-transcript resources shared across all questions."""
    chunks: list
    meeting_start_ts: str
    index: Any
    speaker_registry: dict
    speaker_index: Any
    meeting_duration_min: float


# Only the HuggingFace embedding model is cached — it loads weights onto GPU
# and is expensive to reload. OllamaLLM (HTTP client) and all metric instances
# are created fresh per call so they bind to the thread's current event loop
# and avoid "Event loop is closed" errors from stale async client references.
_judge_embeddings = None


def _get_embeddings():
    global _judge_embeddings
    if _judge_embeddings is None:
        _judge_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )
    return _judge_embeddings


def _run_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    """Synchronous RAGAS evaluation — called inside run_in_executor.

    A fresh event loop is created for the thread so RAGAS async internals have
    a live loop to run on. OllamaLLM and metric instances are also created fresh
    each call so their internal async HTTP clients bind to this loop rather than
    a previously closed one.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        judge_llm = LangchainLLMWrapper(
            OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
        )
        judge_embeddings = _get_embeddings()

        metrics = [
            Faithfulness(llm=judge_llm),
            ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            _LLMContextPrecisionWithoutReference(llm=judge_llm),
            _ContextRelevance(llm=judge_llm),
        ]

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
        ])
        result = evaluate(dataset=dataset, metrics=metrics)
        row = result.to_pandas().iloc[0]
        return {
            "faithfulness": round(float(row["faithfulness"]), 3),
            "answer_relevancy": round(float(row["answer_relevancy"]), 3),
            "context_precision": round(float(row["llm_context_precision_without_reference"]), 3),
            "context_relevance": round(float(row["nv_context_relevance"]), 3),
        }
    finally:
        loop.close()
        asyncio.set_event_loop(None)


async def setup_transcript(transcript: str) -> TranscriptContext:
    """
    Build all transcript-level resources once.
    Called once per evaluation run before looping over questions.
    """
    ws = _MockWebSocket()
    chunks, meeting_start_ts, index = await get_or_build_vectorstore(transcript, ws)
    speaker_registry = build_speaker_registry(chunks)
    speaker_index = build_speaker_index(speaker_registry)
    meeting_duration_min = max(c["metadata"]["end_min"] for c in chunks)
    return TranscriptContext(
        chunks=chunks,
        meeting_start_ts=meeting_start_ts,
        index=index,
        speaker_registry=speaker_registry,
        speaker_index=speaker_index,
        meeting_duration_min=meeting_duration_min,
    )


async def evaluate_question(ctx: TranscriptContext, query: str) -> dict:
    """
    Run retrieval + LLM + RAGAS for a single question using pre-built transcript resources.
    Raises on pipeline errors so the caller can send an error message to the client.
    """
    ws = _MockWebSocket()

    speaker_names, query_names = extract_speakers_from_text(query, ctx.speaker_index)

    if speaker_names:
        filtered_query = remove_matched_speakers(query, query_names)
        doc = nlp(filtered_query)
    else:
        doc = nlp(query)

    mentioned_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    is_synthesis = _detect_synthesis(query)

    result = await search_chunks_with_temporal_and_speaker(
        query=query,
        all_chunks=ctx.chunks,
        index=ctx.index,
        meeting_duration_min=ctx.meeting_duration_min,
        meeting_start_ts=ctx.meeting_start_ts,
        top_k=5,
        speaker_names=speaker_names,
        mentioned_names=mentioned_names,
        websocket=ws,
        is_synthesis=is_synthesis,
    )

    if result is None:
        raise ValueError("No relevant chunks retrieved for this query.")

    docs, _, has_temporal_intent = result
    has_speaker_filter = bool(speaker_names or mentioned_names)
    speaker_label = ", ".join(speaker_names or mentioned_names) if has_speaker_filter else None

    prompt = build_prompt(
        context="\n\n".join(docs),
        query=query,
        has_temporal_intent=has_temporal_intent,
        has_speaker_filter=has_speaker_filter,
        is_synthesis=is_synthesis,
        speaker_label=speaker_label,
    )

    eval_llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, stream=False)
    answer = eval_llm.invoke(prompt)

    scores = await asyncio.get_event_loop().run_in_executor(
        None, partial(_run_ragas, query, answer, docs)
    )

    return {
        "question": query,
        "answer": answer,
        "intent": {
            "temporal": has_temporal_intent,
            "speaker": has_speaker_filter,
            "synthesis": is_synthesis,
        },
        **scores,
    }
