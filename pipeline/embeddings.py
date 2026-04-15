import asyncio
import hashlib
from collections import OrderedDict
from typing import List, Dict, Tuple

import faiss
import numpy as np
from fastapi import WebSocket

from core.models import model
from pipeline.transcript import parse_transcript
from pipeline.chunking import speaker_aware_chunking

_CACHE_MAX_SIZE = 5

# LRU cache: transcript hash -> (chunks, meeting_start_ts, index)
# OrderedDict preserves insertion order; we move accessed keys to the end
# and evict from the front when the cache is full.
_vectorstore_cache: OrderedDict[str, Tuple] = OrderedDict()

# Per-key locks: prevents duplicate builds when two requests arrive
# for the same transcript before the first build completes.
_build_locks: Dict[str, asyncio.Lock] = {}


def embed_chunks_local(chunks: List[Dict]) -> List[Dict]:
    """Embed each chunk's text and store the vector in chunk['embedding']."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def build_faiss_index_local(chunks: List[Dict]) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from pre-embedded chunks."""
    dim = len(chunks[0]["embedding"])
    index = faiss.IndexFlatIP(dim)
    embeddings = np.array([c["embedding"] for c in chunks])
    index.add(embeddings)
    return index


async def build_vectorstore_from_text(
    meeting_text: str,
    websocket: WebSocket,
) -> Tuple[List[Dict], str, faiss.IndexFlatIP]:
    """Parse transcript, chunk, embed, and build FAISS index."""
    df = parse_transcript(meeting_text)
    if df.empty:
        raise ValueError("Transcript parsing failed or empty transcript.")

    meeting_start_ts = df.iloc[0]["timestamp"]

    chunks = speaker_aware_chunking(df, max_chars=1500, overlap_turns=1)
    chunks = embed_chunks_local(chunks)
    index = build_faiss_index_local(chunks)

    return chunks, meeting_start_ts, index


async def get_or_build_vectorstore(
    meeting_text: str,
    websocket: WebSocket,
) -> Tuple[List[Dict], str, faiss.IndexFlatIP]:
    """Return cached vectorstore if transcript unchanged, otherwise build and cache it.

    Uses an LRU eviction policy (max _CACHE_MAX_SIZE entries) and a per-key
    asyncio.Lock to prevent duplicate builds for the same transcript.
    """
    key = hashlib.md5(meeting_text.encode()).hexdigest()

    if key not in _build_locks:
        _build_locks[key] = asyncio.Lock()

    async with _build_locks[key]:
        if key in _vectorstore_cache:
            # Move to end to mark as most recently used
            _vectorstore_cache.move_to_end(key)
            await websocket.send_text("\n\n-> Using cached transcript index")
        else:
            await websocket.send_text("\n\n-> Building transcript index (first query for this transcript)...")
            _vectorstore_cache[key] = await build_vectorstore_from_text(meeting_text, websocket)
            _vectorstore_cache.move_to_end(key)

            # Evict least recently used entry if over the size limit
            if len(_vectorstore_cache) > _CACHE_MAX_SIZE:
                evicted_key, _ = _vectorstore_cache.popitem(last=False)
                _build_locks.pop(evicted_key, None)

    return _vectorstore_cache[key]
