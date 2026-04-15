import re
from typing import List, Tuple

from rapidfuzz import fuzz

from core.models import nlp


def normalize_name(name: str) -> str:
    return name.lower().strip()


def name_tokens(name: str):
    return set(normalize_name(name).split())


def speaker_matches(query_speaker: str, chunk_speaker: str) -> bool:
    """Returns True if all query speaker tokens are present in chunk speaker tokens."""
    q_tokens = name_tokens(query_speaker)
    c_tokens = name_tokens(chunk_speaker)
    return q_tokens.issubset(c_tokens)


def chunk_has_speaker(chunk_speakers, query_speakers) -> bool:
    for qs in query_speakers:
        for cs in chunk_speakers:
            if speaker_matches(qs, cs):
                return True
    return False


def extract_mentioned_names(text: str, speakers: List[str]) -> List[str]:
    """Extract PERSON entities from text that are not part of the known speaker list."""
    doc = nlp(text)
    speaker_tokens = set()
    for speaker in speakers:
        for token in re.findall(r'[a-z]+', speaker.lower()):
            if len(token) >= 2:
                speaker_tokens.add(token)

    names = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_tokens = set(re.findall(r'[a-z]+', ent.text.lower()))
            if not ent_tokens & speaker_tokens:
                names.add(ent.text)
    return sorted(names)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_speaker_registry(chunks) -> List[str]:
    speakers = set()
    for c in chunks:
        if "speakers" in c["metadata"]:
            for name in c["metadata"]["speakers"]:
                speakers.add(name.strip())
    return list(speakers)


def build_speaker_index(speakers: List[str]) -> dict:
    index = {}
    for speaker in speakers:
        clean_speaker = speaker.split("|")[0].strip()
        norm_full = normalize(clean_speaker)
        parts = norm_full.split()

        index[norm_full] = speaker
        for part in parts:
            if len(part) > 2:
                index[part] = speaker
    return index


def extract_speakers_from_text(
    text: str, speaker_index: dict, threshold: int = 80
) -> Tuple[List[str], List[str]]:
    text_norm = normalize(text)
    matched = set()
    query_names = set()
    for key in speaker_index.keys():
        score = fuzz.partial_ratio(key, text_norm)
        if score >= threshold:
            matched.add(speaker_index[key])
            query_names.add(key)
    return list(matched), list(query_names)


def remove_matched_speakers(text: str, matched_speakers: List[str]) -> str:
    cleaned_text = text
    for speaker in matched_speakers:
        base_name = speaker.split("|")[0].strip()
        cleaned_text = re.sub(base_name, "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text
