import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# Indexing
EXTENSIONS = [".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rs", ".cpp", ".c", ".h"]
CHUNK_MAX_TOKENS = 512          # chunks larger than this get split at method boundaries
EMBED_BATCH_SIZE = 64           
EMBED_MODEL = "models/gemini-embedding-2-preview"

# Retrieval
TOP_K = 6                       # number of chunks returned to the LLM
BM25_TOP_K = 20                 # candidates from BM25 before RRF merge
VECTOR_TOP_K = 20               # candidates from vector search before RRF merge
RRF_K = 60                      # RRF constant (standard is 60)
USE_RERANKER = False            # set True to enable cross-encoder reranker

# Generation
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
STREAM = True

# ChromaDB
COLLECTION_NAME = "codebase_chunks_local" #codebase_chunks_local or codebase_chunks for Google API