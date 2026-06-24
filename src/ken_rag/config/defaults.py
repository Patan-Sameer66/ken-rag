"""Default constants for ken-rag.

These are the canonical values imported by Settings.default() and the loader.
No env reads or I/O here — pure constants.
"""

# Retrieval
DEFAULT_K: int = 5

# Generation
NUM_CTX: int = 8192

# Embedding
EMBED_DIM: int = 768
EMBEDDER_NAME: str = "nomic-embed-text"

# LLM
DEFAULT_LLM: str = "qwen2.5:3b"

# Batching
BATCH_SIZE: int = 64

# Prose chunking
PROSE_MIN_TOK: int = 300
PROSE_MAX_TOK: int = 800
PROSE_OVERLAP: float = 0.12

# Ollama
OLLAMA_URL: str = "http://localhost:11434"
TIMEOUT_S: int = 120
