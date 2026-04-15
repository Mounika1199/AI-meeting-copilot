"""
Evaluate LLM output using RAGAS Faithfulness and ResponseRelevancy (answer relevancy).

Requires: pip install ragas langchain-huggingface

Run from the project root:
    python -m eval.evaluate
"""

import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from config import OLLAMA_URL, LLM_MODEL, EMBEDDING_MODEL
from core.models import model, nlp
from pipeline.embeddings import get_or_build_vectorstore
from pipeline.retrieval import search_chunks_with_temporal_and_speaker
from pipeline.prompt import build_prompt
from utils.speaker import (
    build_speaker_registry,
    build_speaker_index,
    extract_speakers_from_text,
    remove_matched_speakers,
)
from eval.test_cases import TEST_CASES

# Synthesis intent phrases — mirrors query_meeting() in retrieval.py
_SYNTHESIS_PHRASES = {
    "action item", "action items", "open question", "open questions",
    "next step", "next steps", "follow up", "follow-up", "key decision",
    "key decisions", "decision", "decisions", "summary", "summarize",
    "main point", "main points", "what was decided", "what did we decide",
    "todo", "to do", "to-do", "everything", "all", "entire", "whole",
}


class _MockWebSocket:
    """Absorbs pipeline send_text calls so evaluation runs without a real WS."""
    async def send_text(self, _: str):
        pass


def _detect_synthesis(query: str) -> bool:
    q = query.lower()
    return any(re.search(r"\b" + re.escape(p) + r"\b", q) for p in _SYNTHESIS_PHRASES)


async def _run_single(transcript: str, query: str) -> dict | None:
    ws = _MockWebSocket()

    # Build / retrieve FAISS index
    chunks, meeting_start_ts, index = await get_or_build_vectorstore(transcript, ws)

    # Speaker extraction
    speaker_registry = build_speaker_registry(chunks)
    speaker_index = build_speaker_index(speaker_registry)
    speaker_names, query_names = extract_speakers_from_text(query, speaker_index)

    if speaker_names:
        filtered_query = remove_matched_speakers(query, query_names)
        doc = nlp(filtered_query)
    else:
        doc = nlp(query)

    mentioned_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    meeting_duration_min = max(c["metadata"]["end_min"] for c in chunks)
    is_synthesis = _detect_synthesis(query)

    # Retrieve relevant chunks
    result = await search_chunks_with_temporal_and_speaker(
        query=query,
        all_chunks=chunks,
        index=index,
        meeting_duration_min=meeting_duration_min,
        meeting_start_ts=meeting_start_ts,
        top_k=5,
        speaker_names=speaker_names,
        mentioned_names=mentioned_names,
        websocket=ws,
        is_synthesis=is_synthesis,
    )

    if result is None:
        print(f"  [skip] No chunks retrieved for: {query[:60]}")
        return None

    docs, _, has_temporal_intent = result
    has_speaker_filter = bool(speaker_names or mentioned_names)

    # Build prompt and get full LLM answer (non-streaming for eval)
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

    return {
        "question": query,
        "answer": answer,
        "contexts": docs,  # list of retrieved chunk strings
    }


async def _collect_results() -> list[dict]:
    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['query'][:70]}")
        r = await _run_single(tc["transcript"], tc["query"])
        if r:
            results.append(r)
    return results


def main():
    print("=== Collecting LLM responses ===")
    results = asyncio.run(_collect_results())

    if not results:
        print("No results collected. Check your test cases.")
        return

    print(f"\n=== Running RAGAS on {len(results)} sample(s) ===")

    # Wire Ollama as the judge LLM (temperature=0 for deterministic judgements)
    judge_llm = LangchainLLMWrapper(
        OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
    )
    # ResponseRelevancy also needs an embedding model to measure semantic similarity
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    faithfulness_metric = Faithfulness(llm=judge_llm)
    relevancy_metric = ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings)

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )
        for r in results
    ])

    scores = evaluate(dataset=dataset, metrics=[faithfulness_metric, relevancy_metric])
    df = scores.to_pandas()

    print("\n=== Results ===")
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))
    print(f"\nMean faithfulness:     {df['faithfulness'].mean():.3f}")
    print(f"Mean answer_relevancy: {df['answer_relevancy'].mean():.3f}")


if __name__ == "__main__":
    main()
