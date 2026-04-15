import os

from config import (
    CUDA_VISIBLE_DEVICES,
    OLLAMA_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPEAT_PENALTY,
    LLM_MAX_TOKENS,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    SPACY_MODEL,
)

# Must be set before GPU-aware libraries are imported
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

import logging
import spacy
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from langchain_ollama.llms import OllamaLLM

# spaCy NLP pipeline
nlp = spacy.load(SPACY_MODEL)

# Sentence embedding model
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"Embedding model loaded: {EMBEDDING_MODEL} | Dimensions: {model.get_sentence_embedding_dimension()}")

# Cross-encoder reranker
reranker = FlagReranker(RERANKER_MODEL)

# LLM
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    top_p=LLM_TOP_P,
    repeat_penalty=LLM_REPEAT_PENALTY,
    num_predict=LLM_MAX_TOKENS,
    base_url=OLLAMA_URL,
    stream=True,
)
