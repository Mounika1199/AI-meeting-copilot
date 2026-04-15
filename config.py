import os
from dotenv import load_dotenv

load_dotenv()

# GPU
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "8,9")

# Ollama / LLM
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b-20k")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Embedding / reranker models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_md")

# Context / retrieval
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
# bge-reranker scores are raw logits: >0 relevant, < -4 likely irrelevant
RERANKER_RELEVANCE_THRESHOLD = float(os.getenv("RERANKER_RELEVANCE_THRESHOLD", "-3.0"))
# Per-doc threshold for factual/speaker queries — chunks below this are dropped
FACTUAL_SCORE_THRESHOLD = float(os.getenv("FACTUAL_SCORE_THRESHOLD", "-5.0"))
# Max chunks considered after temporal filtering before speaker filtering / reranking
TEMPORAL_TOP_K = int(os.getenv("TEMPORAL_TOP_K", "10"))

# Server
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8018"))
