import os

from dotenv import load_dotenv

load_dotenv()

# =========================
# Ollama
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# =========================
# MySQL
# =========================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "knowledge_base")

# =========================
# Qdrant
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "kb_chunks")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "dense")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# =========================
# RAG 默认参数
# =========================
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 100))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 3))

# =========================
# Hybrid Retrieval / 检索参数
# =========================

# 两路召回候选数
HYBRID_VECTOR_TOP_K = int(os.getenv("HYBRID_VECTOR_TOP_K", 20))
HYBRID_KEYWORD_TOP_K = int(os.getenv("HYBRID_KEYWORD_TOP_K", 20))

# DB lexical 召回取数
HYBRID_LEXICAL_FETCH_K = int(os.getenv("HYBRID_LEXICAL_FETCH_K", 120))
HYBRID_BOOLEAN_FETCH_K = int(os.getenv("HYBRID_BOOLEAN_FETCH_K", 80))
HYBRID_HYDRATE_TOP_K = int(os.getenv("HYBRID_HYDRATE_TOP_K", 160))

# 最终重排权重
RETRIEVAL_WEIGHT_EMBEDDING = float(
    os.getenv("RETRIEVAL_WEIGHT_EMBEDDING", 0.45)
)
RETRIEVAL_WEIGHT_KEYWORD = float(
    os.getenv("RETRIEVAL_WEIGHT_KEYWORD", 0.20)
)
RETRIEVAL_WEIGHT_TITLE = float(
    os.getenv("RETRIEVAL_WEIGHT_TITLE", 0.15)
)
RETRIEVAL_WEIGHT_SECTION = float(
    os.getenv("RETRIEVAL_WEIGHT_SECTION", 0.10)
)
RETRIEVAL_WEIGHT_BM25 = float(
    os.getenv("RETRIEVAL_WEIGHT_BM25", 0.10)
)

# 关键词打分细节
KEYWORD_EXACT_MATCH_WEIGHT = float(
    os.getenv("KEYWORD_EXACT_MATCH_WEIGHT", 1.0)
)
KEYWORD_SUBSTRING_MATCH_WEIGHT = float(
    os.getenv("KEYWORD_SUBSTRING_MATCH_WEIGHT", 0.55)
)
TITLE_MATCH_WEIGHT = float(
    os.getenv("TITLE_MATCH_WEIGHT", 0.8)
)
SECTION_MATCH_WEIGHT = float(
    os.getenv("SECTION_MATCH_WEIGHT", 0.6)
)

# 去重 / 多样性控制
RETRIEVAL_DEDUP_SIM_THRESHOLD = float(
    os.getenv("RETRIEVAL_DEDUP_SIM_THRESHOLD", 0.82)
)
RETRIEVAL_MAX_SAME_SECTION = int(
    os.getenv("RETRIEVAL_MAX_SAME_SECTION", 2)
)

# 邻接 chunk 扩展
RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION = os.getenv(
    "RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION", "true"
).lower() in {"1", "true", "yes", "on"}
RETRIEVAL_NEIGHBOR_WINDOW = int(
    os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", 1)
)

# =========================
# Query 预处理
# =========================
QUERY_MIN_TERM_LEN = int(os.getenv("QUERY_MIN_TERM_LEN", 2))

QUERY_STOPWORDS = {
    "什么",
    "是什么",
    "怎么",
    "如何",
    "为什么",
    "为何",
    "怎样",
    "一下",
    "一个",
    "这个",
    "那个",
    "这些",
    "那些",
    "以及",
    "和",
    "与",
    "或",
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "呀",
    "吧",
    "在",
    "中",
    "里",
    "上",
    "下",
    "对",
    "把",
    "被",
    "就",
    "都",
    "也",
    "还",
    "是",
    "有",
    "用",
    "讲",
    "说",
    "介绍",
    "区别",
    "优点",
    "缺点",
    "优缺点",
}

# =========================
# 其他
# =========================
DATA_DIR = os.getenv("DATA_DIR", "data")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))