from typing import List, Dict

import pandas as pd

from utils.speaker import extract_mentioned_names
from utils.temporal import timestamp_to_minutes


def speaker_aware_chunking(
    df: pd.DataFrame,
    max_chars: int = 2500,
    overlap_turns: int = 1,
) -> List[Dict]:
    """
    Split transcript DataFrame into overlapping chunks with metadata.

    Returns a list of dicts, each with:
        - text: combined speaker-turn text
        - metadata: speakers, mentioned_names, start/end times, start/end minutes
    """
    chunks = []
    current_chunk = []
    current_len = 0
    meeting_start_min = timestamp_to_minutes(df.iloc[0]["timestamp"])

    for _, row in df.iterrows():
        turn_text = f"{row.speaker} ({row.timestamp}): {row.text}"
        turn_len = len(turn_text)

        if current_len + turn_len > max_chars and current_chunk:
            chunk_text = "\n".join(t["text"] for t in current_chunk)
            speakers = list(set(t["speaker"] for t in current_chunk))
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "speakers": speakers,
                    "mentioned_names": extract_mentioned_names(chunk_text, speakers),
                    "start_time": current_chunk[0]["timestamp"],
                    "end_time": current_chunk[-1]["timestamp"],
                    "start_min": timestamp_to_minutes(current_chunk[0]["timestamp"]) - meeting_start_min,
                    "end_min": timestamp_to_minutes(current_chunk[-1]["timestamp"]) - meeting_start_min,
                }
            })
            current_chunk = current_chunk[-overlap_turns:]
            current_len = sum(len(t["text"]) for t in current_chunk)

        current_chunk.append({
            "speaker": row.speaker,
            "timestamp": row.timestamp,
            "text": turn_text,
        })
        current_len += turn_len

    # Final chunk
    if current_chunk:
        chunk_text = "\n".join(t["text"] for t in current_chunk)
        speakers = list(set(t["speaker"] for t in current_chunk))
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "speakers": speakers,
                "mentioned_names": extract_mentioned_names(chunk_text, speakers),
                "start_time": current_chunk[0]["timestamp"],
                "end_time": current_chunk[-1]["timestamp"],
                "start_min": timestamp_to_minutes(current_chunk[0]["timestamp"]) - meeting_start_min,
                "end_min": timestamp_to_minutes(current_chunk[-1]["timestamp"]) - meeting_start_min,
            }
        })

    return chunks
