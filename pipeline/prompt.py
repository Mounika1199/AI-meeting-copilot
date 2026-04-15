from typing import Optional


def build_prompt(
    context: str,
    query: str,
    has_temporal_intent: bool,
    has_speaker_filter: bool,
    is_synthesis: bool,
    speaker_label: Optional[str] = None,
) -> str:
    """Return an intent-aware prompt for the LLM based on the active query filters."""

    if has_temporal_intent and has_speaker_filter and is_synthesis:
        # Triple Hybrid: Temporal + Speaker + Synthesis
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts from a specific time window of the meeting, filtered to {speaker_label}.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer the question directly and specifically about {speaker_label} within that time period.
- Use bullet points or numbered lists for clarity.
- Do not include other speakers or content outside the time window.
- Do not make up information not present in the context."""

    if has_temporal_intent and has_speaker_filter:
        # Hybrid: Temporal + Speaker
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts from a specific time window of the meeting, filtered to {speaker_label}.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer the question directly and concisely about {speaker_label} within that time period.
- If {speaker_label} did not say anything relevant in that time window, say so explicitly.
- Do not make up information not present in the context."""

    if has_temporal_intent and is_synthesis:
        # Hybrid: Temporal + Synthesis
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts from a specific time window of the meeting.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer the question directly. Do not add unrequested sections.
- Use bullet points or numbered lists for clarity.
- Do not include information from outside the specified time window.
- Do not make up information not present in the context."""

    if has_speaker_filter and is_synthesis:
        # Hybrid: Speaker + Synthesis
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts attributed to {speaker_label}.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer the question directly and specifically about {speaker_label}.
- Use bullet points or numbered lists for clarity.
- Do not mix in other speakers.
- Do not make up information not present in the context."""

    if has_temporal_intent:
        # Pure Temporal
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts from a specific time window of the meeting.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer based only on what was discussed during the specified time period.
- Be precise and reference the discussion directly.
- If the topic was not discussed in that time window, say so explicitly.
- Do not make up information not present in the context."""

    if has_speaker_filter:
        # Pure Speaker (factual question scoped to a speaker)
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts relevant to {speaker_label}.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer the question directly and concisely — do not summarize everything {speaker_label} said.
- Scope your answer to {speaker_label} only where relevant to the question.
- If the answer is not in the context, say so clearly.
- Do not make up information not present in the context."""

    if is_synthesis:
        # Pure Synthesis
        return f"""You are an intelligent Meeting Copilot.
The context below contains transcript excerpts from the meeting.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Answer the question directly and specifically — do not produce a general meeting summary unless explicitly asked.
- Use bullet points or numbered lists for clarity.
- Include only what directly answers the question. Do not add unrequested sections.
- Do not make up information not present in the context."""

    # Pure Factual / Semantic
    return f"""You are an intelligent Meeting Copilot.
The context below contains the most relevant transcript excerpts for your question.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Give a direct, concise answer based on the context.
- Include only what is necessary to answer the question.
- If the answer is not in the context, say so clearly.
- Do not make up information not present in the context."""
