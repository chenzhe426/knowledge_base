import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Ollama
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# =========================
# MySQL
# =========================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "zyj123456")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "knowledge_base")

# =========================
# Chunk 默认参数
# =========================
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 100))

# =========================
# Retrieval 默认参数
# =========================
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))

# 向量粗召回候选数
DEFAULT_RETRIEVE_CANDIDATES = int(os.getenv("DEFAULT_RETRIEVE_CANDIDATES", 30))

# 重排前保留的候选数（如果你后面在 services.py 里单独使用）
DEFAULT_RERANK_CANDIDATES = int(os.getenv("DEFAULT_RERANK_CANDIDATES", 20))

# 最终喂给大模型的上下文长度上限（粗略 token）
DEFAULT_MAX_CONTEXT_TOKENS = int(os.getenv("DEFAULT_MAX_CONTEXT_TOKENS", 1800))

# 检索命中 chunk 后，向前/向后补几个相邻 chunk
DEFAULT_NEIGHBOR_RADIUS = int(os.getenv("DEFAULT_NEIGHBOR_RADIUS", 1))

# 去重/多样性控制：同一文档最多保留多少个 chunk
DEFAULT_PER_DOC_LIMIT = int(os.getenv("DEFAULT_PER_DOC_LIMIT", 4))

# 近重复判定阈值
DEFAULT_NEAR_DUP_THRESHOLD = float(os.getenv("DEFAULT_NEAR_DUP_THRESHOLD", 0.82))

# =========================
# Hybrid Retrieval 权重
# =========================
HYBRID_WEIGHT_EMBEDDING = float(os.getenv("HYBRID_WEIGHT_EMBEDDING", 0.50))
HYBRID_WEIGHT_LEXICAL = float(os.getenv("HYBRID_WEIGHT_LEXICAL", 0.28))
HYBRID_WEIGHT_METADATA = float(os.getenv("HYBRID_WEIGHT_METADATA", 0.22))