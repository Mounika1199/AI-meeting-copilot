import re
from typing import Optional, Dict

from fastapi import WebSocket
from rapidfuzz import fuzz
from word2number import w2n

from core.models import nlp


# --------------------------------------------------------
# Constants
# --------------------------------------------------------

TIME_REGEX = r"\b([01]?\d|2[0-3]):([0-5]\d)\b"

FUZZY_AROUND = {"around", "about", "roughly", "approximately", "near"}
FUZZY_BEFORE = {"just before", "shortly before"}
FUZZY_AFTER = {"just after", "shortly after"}

FUZZY_WINDOW = 5  # minutes on each side

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "ninety"
}

BEGINNING_KEYWORDS = {
    "first", "initial", "early", "beginning", "start", "starting", "initially"
}
ENDING_KEYWORDS = {
    "last", "final", "end", "ending"
}


# --------------------------------------------------------
# Utilities
# --------------------------------------------------------

def timestamp_to_minutes(ts: str) -> int:
    """Convert HH:MM timestamp to total minutes."""
    h, m = map(int, ts.split(":"))
    return h * 60 + m


def extract_timestamps(text: str):
    """Return list of timestamps found in text as total minutes."""
    matches = re.findall(TIME_REGEX, text)
    return [int(h) * 60 + int(m) for h, m in matches]


def extract_clock_times(text: str):
    matches = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", text)
    return matches


def chunk_in_time_window(chunk: dict, start_min: int, end_min: int) -> bool:
    """Returns True if chunk overlaps with [start_min, end_min]."""
    c_start = chunk["metadata"]["start_min"]
    c_end = chunk["metadata"]["end_min"]
    return c_end >= start_min and c_start <= end_min


def fuzzy_contains(token: str, keywords, threshold: int = 85) -> bool:
    for kw in keywords:
        if fuzz.ratio(token, kw) >= threshold:
            return True
    return False


# --------------------------------------------------------
# Temporal Intent Detection
# --------------------------------------------------------

async def detect_temporal_intent_spacy(
    query: str,
    meeting_start_ts: str,
    meeting_duration_min: int,
    websocket: WebSocket,
    default_window: int = 12,
) -> Optional[Dict]:
    """
    Detect temporal intent from query.

    Returns:
        - {'start_min': X, 'end_min': Y}  for range/beginning queries
        - {'last_minutes': X}              for ending queries
        - None                             if no temporal intent detected
    """
    doc = nlp(query.lower())
    tokens = [tok.text.lower() for tok in doc]
    text = " ".join(tokens)

    # --------------------------------------------------
    # 0. Detect absolute clock times
    # --------------------------------------------------
    clock_times = extract_clock_times(text)

    if clock_times:
        await websocket.send_text("\n\n-> User mentioned clock in the Query")
        meeting_start_min_absolute = timestamp_to_minutes(meeting_start_ts)

        relative_times = [
            timestamp_to_minutes(ts) - meeting_start_min_absolute
            for ts in clock_times
        ]
        raw_relative_times = relative_times.copy()

        # Clamp to meeting bounds
        relative_times = [
            max(0, min(meeting_duration_min, t)) for t in relative_times
        ]

        # BETWEEN / FROM X TO Y
        if len(relative_times) >= 2:
            start_min, end_min = sorted(relative_times[:2])
            await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
            return {"start_min": start_min, "end_min": end_min}

        t = relative_times[0]

        if any(word in text for word in FUZZY_AROUND):
            start_min = max(0, t - FUZZY_WINDOW)
            end_min = min(meeting_duration_min, t + FUZZY_WINDOW)
            await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
            return {"start_min": start_min, "end_min": end_min}

        if any(word in text for word in FUZZY_BEFORE):
            start_min = max(0, t - FUZZY_WINDOW)
            end_min = t
            await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
            return {"start_min": start_min, "end_min": end_min}

        if any(word in text for word in FUZZY_AFTER):
            start_min = t
            end_min = min(meeting_duration_min, t + FUZZY_WINDOW)
            await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
            return {"start_min": start_min, "end_min": end_min}

        if "before" in text or "until" in text:
            start_min = 0
            end_min = t
            await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
            return {"start_min": start_min, "end_min": end_min}

        if "after" in text or "since" in text or "from" in text:
            start_min = t
            end_min = meeting_duration_min
            await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
            return {"start_min": t, "end_min": meeting_duration_min}

        # Exact timestamp fallback — return None window if outside meeting
        t = raw_relative_times[0]
        if t < 0 or t > meeting_duration_min:
            return {"start_min": -1, "end_min": -1}

        start_min = max(0, t - default_window // 2)
        end_min = min(meeting_duration_min, t + default_window // 2)
        await websocket.send_text(f'\n\n-> start_min: {start_min}, end_min: {end_min}')
        return {"start_min": start_min, "end_min": end_min}

    # --------------------------------------------------
    # 1. Reject calendar-unit time references (not meeting-relative)
    # --------------------------------------------------
    # "last 2 weeks", "past few days", "over the last month" etc. refer to
    # real-world time, not a position in the meeting timeline.
    if re.search(r"\b\d*\s*(week|day|month|year|hour)s?\b", text):
        return None

    # --------------------------------------------------
    # 2. Extract duration (digit or word form, minutes only)
    # --------------------------------------------------
    duration_min = None

    digit_match = re.search(r"\b(\d+)\s*(minutes?|mins?)\b", text)
    if digit_match:
        duration_min = int(digit_match.group(1))
    else:
        for i, tok in enumerate(tokens):
            if tok.startswith("minute"):
                number_tokens = []
                j = i - 1
                while j >= 0:
                    if tokens[j] in NUMBER_WORDS:
                        number_tokens.insert(0, tokens[j])
                    elif tokens[j] in {"a", "an"}:
                        pass
                    else:
                        break
                    j -= 1

                if number_tokens:
                    try:
                        duration_min = w2n.word_to_num(" ".join(number_tokens))
                    except Exception:
                        duration_min = default_window
                    break

    if duration_min is None:
        duration_min = default_window

    # --------------------------------------------------
    # 2. Detect beginning / ending intent
    # --------------------------------------------------
    has_first = any(kw in text for kw in [
        "at the beginning", "in the beginning", "at the start", "early on"
    ])
    has_last = any(kw in text for kw in [
        "at the end", "towards the end", "wrap up", "closing"
    ])

    if not has_first:
        has_first = any(tok in BEGINNING_KEYWORDS for tok in tokens)
    if not has_last:
        has_last = any(tok in ENDING_KEYWORDS for tok in tokens)

    # --------------------------------------------------
    # 3. Construct output
    # --------------------------------------------------
    if has_last and duration_min:
        await websocket.send_text(f'\n\n-> last_minutes: {duration_min}')
        return {"last_minutes": duration_min}

    if has_first and duration_min:
        await websocket.send_text(f'\n\n-> start_min: {0}, end_min: {duration_min}')
        return {"start_min": 0, "end_min": duration_min}

    if has_first:
        await websocket.send_text(f'\n\n-> start_min: {0}, end_min: {default_window}')
        return {"start_min": 0, "end_min": default_window}

    if has_last:
        start = max(0, meeting_duration_min - default_window)
        await websocket.send_text(f'\n\n-> start_min: {start}, end_min: {meeting_duration_min}')
        return {"start_min": start, "end_min": meeting_duration_min}

    return None
