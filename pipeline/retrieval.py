import logging
import re
from typing import List, Optional, Tuple

from fastapi import WebSocket

from config import MAX_CONTEXT_CHARS, RERANKER_RELEVANCE_THRESHOLD, TEMPORAL_TOP_K, FACTUAL_SCORE_THRESHOLD
from core.models import model, llm, nlp
from pipeline.embeddings import get_or_build_vectorstore
from pipeline.reranker import rerank
from utils.speaker import (
    build_speaker_registry,
    build_speaker_index,
    remove_matched_speakers,
    extract_speakers_from_text,
    chunk_has_speaker,
)
from utils.temporal import (
    detect_temporal_intent_spacy,
    chunk_in_time_window,
)
from pipeline.prompt import build_prompt



# --------------------------------------------------------
# Retrieval pipeline
# --------------------------------------------------------

async def search_chunks_with_temporal_and_speaker(
    query: str,
    all_chunks: list,
    index,
    meeting_duration_min: int,
    meeting_start_ts: str,
    websocket: WebSocket,
    top_k: int = 5,
    speaker_names: Optional[List[str]] = None,
    mentioned_names: Optional[List[str]] = None,
    is_synthesis: bool = False,
) -> Optional[Tuple[List[str], float, bool]]:
    """
    Full retrieval pipeline:
      1. Temporal intent detection + prefiltering
      2. Semantic (FAISS) search
      3. Speaker filtering
      4. Cross-encoder reranking
    """

    # --------------------------------------------------
    # 1. Temporal intent detection
    # --------------------------------------------------
    temporal_intent = await detect_temporal_intent_spacy(
        query,
        meeting_start_ts=meeting_start_ts,
        meeting_duration_min=meeting_duration_min,
        websocket=websocket,
    )

    candidate_chunks = all_chunks

    # --------------------------------------------------
    # 2. Temporal prefiltering
    # --------------------------------------------------
    if temporal_intent:
        top_k = TEMPORAL_TOP_K
        if "last_minutes" in temporal_intent:
            start_min = max(0, meeting_duration_min - temporal_intent["last_minutes"])
            end_min = meeting_duration_min
        else:
            start_min = temporal_intent["start_min"]
            end_min = temporal_intent["end_min"]

        if start_min == end_min:
            return None

        candidate_chunks = [
            c for c in all_chunks if chunk_in_time_window(c, start_min, end_min)
        ]
        if not candidate_chunks:
            candidate_chunks = all_chunks

        candidate_chunks = candidate_chunks[:top_k]

        retrieved_texts1 = []
        retrieved_texts2 = []

        if speaker_names:
            results = [
                c for c in candidate_chunks
                if chunk_has_speaker(c["metadata"]["speakers"], speaker_names)
            ]
            retrieved_texts1 = [r["text"] for r in results]
            if not retrieved_texts1:
                await websocket.send_text(f"\n\n-> Applying Speaker Filtering based on {speaker_names}")

        if mentioned_names:
            results = [
                c for c in candidate_chunks
                if chunk_has_speaker(c["metadata"]["mentioned_names"], mentioned_names)
            ]
            retrieved_texts2 = [r["text"] for r in results]
            if not retrieved_texts2:
                await websocket.send_text(f"\n\n-> {mentioned_names} Speakers not found in the Transcript")
            else:
                await websocket.send_text(f"\n\n-> {mentioned_names} Speakers found in the Transcript")
                await websocket.send_text(f"\n\n-> Applying Speaker Filtering based on {mentioned_names}")

        retrieved_texts = list(set(retrieved_texts1 + retrieved_texts2))


        if not speaker_names and not mentioned_names:
            temporal_texts = [c["text"] for c in candidate_chunks]
            docs, _, score = rerank(query=query, docs=temporal_texts, top_k=TEMPORAL_TOP_K)
            return docs, score, True

        if not retrieved_texts:
            return None

        docs, _, score = rerank(query=query, docs=retrieved_texts, top_k=TEMPORAL_TOP_K)
        return docs, score, True

    # --------------------------------------------------
    # 3. Encode query for semantic search
    # --------------------------------------------------
    query_emb = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    # --------------------------------------------------
    # 4. Speaker filtering (no temporal intent)
    # --------------------------------------------------
    if speaker_names or mentioned_names:
        top_k = 8 if is_synthesis else 5
        distances, indices = index.search(query_emb, top_k * 3)

        retrieved_texts1 = []
        retrieved_texts2 = []

        if speaker_names:
            results = []
            for idx, _ in zip(indices[0], distances[0]):
                chunk = all_chunks[idx]
                if chunk_has_speaker(chunk["metadata"]["speakers"], speaker_names):
                    results.append(chunk)
                    if len(results) >= top_k:
                        break
            retrieved_texts1 = [r["text"] for r in results]
            if retrieved_texts1:
                await websocket.send_text(f"\n\n-> Applying Speaker Filtering based on {speaker_names}")
            else:
                await websocket.send_text(f"\n\n-> Speaker {speaker_names} not found in the Transcript")

        if mentioned_names:
            results = [
                all_chunks[idx]
                for idx, _ in zip(indices[0], distances[0])
                if chunk_has_speaker(all_chunks[idx]["metadata"]["mentioned_names"], mentioned_names)
            ]
            retrieved_texts2 = [r["text"] for r in results]
            if not retrieved_texts2:
                await websocket.send_text(f"\n\n-> Speaker {mentioned_names} not found in the Transcript")
            else:
                await websocket.send_text(f"\n\n-> Speaker {mentioned_names} found in the Transcript")
                await websocket.send_text(f"\n\n-> Applying Speaker Filtering based on {mentioned_names}")

        retrieved_texts = list(set(retrieved_texts1 + retrieved_texts2))

        if not retrieved_texts:
            return None

        speaker_rerank_top_k = 8 if is_synthesis else 3
        docs, doc_scores, score = rerank(query=query, docs=retrieved_texts, top_k=speaker_rerank_top_k)
        if not is_synthesis:
            filtered = [(d, s) for d, s in zip(docs, doc_scores) if s >= FACTUAL_SCORE_THRESHOLD]
            docs = [d for d, _ in filtered] if filtered else [docs[0]]
        return docs, score, False

    # --------------------------------------------------
    # 5. Pure semantic search (no filters)
    # --------------------------------------------------
    await websocket.send_text("\n\n-> Not detected Temporal and Speaker Intent in the Query")
    # Synthesis queries need a wider FAISS net to surface relevant chunks
    faiss_top_k = 8 if is_synthesis else 5
    distances, indices = index.search(query_emb, faiss_top_k * 3)

    results = [all_chunks[idx] for idx in indices[0]]
    retrieved_texts = [r["text"] for r in results]

    rerank_top_k = 8 if is_synthesis else 3
    docs, doc_scores, score = rerank(query=query, docs=retrieved_texts, top_k=rerank_top_k)
    if not is_synthesis:
        filtered = [(d, s) for d, s in zip(docs, doc_scores) if s >= FACTUAL_SCORE_THRESHOLD]
        docs = [d for d, _ in filtered] if filtered else [docs[0]]
    return docs, score, False


# --------------------------------------------------------
# Main query entry point
# --------------------------------------------------------

async def query_meeting(meeting_text: str, query: str, websocket: WebSocket):
    """Build index from transcript and stream LLM response for the given query."""
    logging.info(f"Received query: {query[:100]}")

    # Step 1: Build FAISS index (cached per transcript)
    chunks, meeting_start_ts, index = await get_or_build_vectorstore(meeting_text, websocket)

    # Step 2: Speaker extraction
    speaker_registry = build_speaker_registry(chunks)
    speaker_index = build_speaker_index(speaker_registry)

    speaker_names, query_names = extract_speakers_from_text(query, speaker_index)
    if speaker_names:
        await websocket.send_text(f"\n\n-> Speakers {speaker_names} found in the Transcript")
        query_filtered_text = remove_matched_speakers(query, query_names)
        doc = nlp(query_filtered_text)
    else:
        doc = nlp(query)

    mentioned_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if mentioned_names:
        await websocket.send_text(
            f"\n\n-> {mentioned_names} is not found in the Speaker registry but detected by NER"
        )

    meeting_duration_min = max(c["metadata"]["end_min"] for c in chunks)

    # Step 3: Detect synthesis intent (questions spanning the full transcript)
    SYNTHESIS_PHRASES = {
        "action item", "action items", "open question", "open questions",
        "next step", "next steps", "follow up", "follow-up", "key decision",
        "key decisions", "decision", "decisions", "summary", "summarize",
        "main point", "main points", "what was decided", "what did we decide",
        "todo", "to do", "to-do", "everything", "all", "entire", "whole",
    }
    query_lower = query.lower()
    is_synthesis = any(
        re.search(r"\b" + re.escape(phrase) + r"\b", query_lower)
        for phrase in SYNTHESIS_PHRASES
    )
    if is_synthesis:
        await websocket.send_text("\n\n-> Synthesis intent detected: widening retrieval across full transcript")

    # Step 4: Retrieve relevant chunks
    results = await search_chunks_with_temporal_and_speaker(
        query=query,
        all_chunks=chunks,
        index=index,
        meeting_duration_min=meeting_duration_min,
        meeting_start_ts=meeting_start_ts,
        top_k=5,
        speaker_names=speaker_names,
        mentioned_names=mentioned_names,
        websocket=websocket,
        is_synthesis=is_synthesis,
    )

    if results is None:
        await websocket.send_text(
            "No relevant content was found in the transcript for your query. "
            "Please try rephrasing or check that the speaker name or time reference exists in the transcript."
        )
        await websocket.send_text("\n\n[End of Response]")
        return

    docs, best_score, has_temporal_intent = results

    has_speaker_filter = bool(speaker_names or mentioned_names)
    if not has_temporal_intent and not is_synthesis and not has_speaker_filter and best_score < RERANKER_RELEVANCE_THRESHOLD:
        await websocket.send_text(
            f"Your question doesn't appear to relate to the meeting content (relevance score: {best_score:.2f}). "
            "Please ask something about what was discussed in the meeting."
        )
        await websocket.send_text("\n\n[End of Response]")
        return
    await websocket.send_text(f'\n\n-> Number of chunks retrieved: {len(docs)}\n\n')
    context = "\n\n".join(docs)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]
        await websocket.send_text(
            f"\n\n-> Warning: Context truncated to {MAX_CONTEXT_CHARS} characters to fit model window."
        )
    if has_temporal_intent:
        await websocket.send_text('\n\n-> Detected temporal intent\n\n')
    if has_speaker_filter:
        await websocket.send_text(f"\n\n-> Detected speaker intent\n\n")
    if is_synthesis:
        await websocket.send_text(f"\n\n-> Detected synthesis intent\n\n")
    if not has_temporal_intent and not has_speaker_filter and not is_synthesis:
        await websocket.send_text(f"\n\n-> Detected factual intent\n\n")
    await websocket.send_text('\n\n-> Sending filtered chunks to the LLM\n\n')

    # Step 4: Build intent-aware prompt and stream LLM response
    speaker_label = ", ".join(speaker_names or mentioned_names) if has_speaker_filter else None
    prompt = build_prompt(
        context=context,
        query=query,
        has_temporal_intent=has_temporal_intent,
        has_speaker_filter=has_speaker_filter,
        is_synthesis=is_synthesis,
        speaker_label=speaker_label,
    )
    
    async for chunk in llm.astream(prompt): 
        await websocket.send_text(chunk)
    await websocket.send_text("\n\n[End of Response]")
