import re

import pandas as pd


def parse_transcript(raw_text: str) -> pd.DataFrame:
    """
    Parses meeting transcripts of the form:

        Markus Klehr  15:52
        Yeah, I tested the new API.

        Anja Seifried  15:53
        Great. Any feedback?
    """
    pattern = r"([\w\s\|\.\-]+)\s+(\d{1,2}:\d{2})\s*\n(.+?)(?=\n[\w\s\|\.\-]+\s+\d{1,2}:\d{2}|\Z)"
    matches = re.findall(pattern, raw_text, flags=re.S)

    data = []
    for speaker, timestamp, text in matches:
        clean_text = re.sub(r'\s+', ' ', text.strip())
        data.append({
            "speaker": speaker.strip().lower(),
            "timestamp": timestamp,
            "text": clean_text,
        })

    return pd.DataFrame(data)
