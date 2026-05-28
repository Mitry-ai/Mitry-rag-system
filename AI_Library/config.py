# config.py
import json
import os
from dataclasses import dataclass

@dataclass
class AIModeConfig:
    """AI模式配置"""
    name: str
    temperature: float
    search_k: int
    description: str

# 三种AI模式配置
AI_MODES = {
    "precise": AIModeConfig(
        name="精准检索",
        temperature=0.1,
        search_k=3,
        description="高准确性，适合事实查询"
    ),
    "balanced": AIModeConfig(
        name="综合模式", 
        temperature=0.3,
        search_k=4,
        description="平衡准确性和创造性"
    ),
    "explorative": AIModeConfig(
        name="探索模式",
        temperature=0.7,
        search_k=5,
        description="高创造性，适合创意生成"
    )
}

# 路径基准，避免依赖当前工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

# 数据库配置
DATABASE_CONFIG = {
    "db_path": os.path.join(BASE_DIR, "users.db"),
    "vector_db_path": os.path.join(BASE_DIR, "chroma_db_local")
}


def chroma_persist_directory(path=None):
    """Return a Chroma-safe relative path from BASE_DIR.

    hnswlib on this Windows environment does not persist index files correctly
    when the absolute path contains spaces/non-ASCII characters.
    """
    target = os.path.abspath(path or DATABASE_CONFIG["vector_db_path"])
    try:
        return os.path.relpath(target, BASE_DIR)
    except ValueError:
        return target


def chroma_collection_metadata():
    return {
        "hnsw:batch_size": max(3, int(os.getenv("AI_CHROMA_HNSW_BATCH_SIZE", "32"))),
        "hnsw:sync_threshold": max(3, int(os.getenv("AI_CHROMA_SYNC_THRESHOLD", "3"))),
    }

# 系统配置
SYSTEM_CONFIG = {
    "max_concurrent_users": 1,
    "default_model": "deepseek-r1",
    "embedding_model": "nomic-embed-text:latest"
}

def _env_text(name, default=""):
    return os.getenv(name, default).strip().strip("'\"")


def _env_flag(name, default="0"):
    return _env_text(name, default).lower() in {"1", "true", "yes", "on"}


# 管理员高风险操作二级密码，从环境变量读取，避免写入代码仓库。
ADMIN_SECONDARY_PASSWORD = _env_text("AI_ADMIN_SECONDARY_PASSWORD", "")


# 问答审计配置（开启后按 JSONL 追加记录）
# 如果不想依赖终端环境变量，可以直接把这里改为 True/False。
AUDIT_ENABLED_DEFAULT = True
AUDIT_INCLUDE_ANSWER_DEFAULT = False
AUDIT_SNIPPET_CHARS_DEFAULT = 140

AUDIT_CONFIG = {
    "enabled": _env_flag("AI_AUDIT_ENABLED", "1" if AUDIT_ENABLED_DEFAULT else "0"),
    "log_path": _env_text(
        "AI_AUDIT_LOG_PATH",
        os.path.join(BASE_DIR, "logs", "qa_trace.jsonl"),
    ),
    "include_answer": _env_flag(
        "AI_AUDIT_INCLUDE_ANSWER",
        "1" if AUDIT_INCLUDE_ANSWER_DEFAULT else "0",
    ),
    "snippet_chars": int(_env_text("AI_AUDIT_SNIPPET_CHARS", str(AUDIT_SNIPPET_CHARS_DEFAULT))),
}


def _load_metadata_filter_from_env():
    raw = os.getenv("AI_METADATA_FILTER_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# 检索配置（通过环境变量切换 dense/hybrid 便于基线对比）
RETRIEVAL_CONFIG = {
    "strategy": os.getenv("AI_RETRIEVAL_STRATEGY", "hybrid").lower(),
    "metadata_filter": _load_metadata_filter_from_env(),
    "reranker": {
        "enabled": os.getenv("AI_RERANK_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"},
        "candidate_k": int(os.getenv("AI_RERANK_CANDIDATE_K", "15")),
        "lexical_weight": float(os.getenv("AI_RERANK_LEXICAL_WEIGHT", "0.7")),
        "prior_weight": float(os.getenv("AI_RERANK_PRIOR_WEIGHT", "0.3")),
    },
    "context_builder": {
        "max_chunks": int(os.getenv("AI_CONTEXT_MAX_CHUNKS", "7")),
        "max_total_chars": int(os.getenv("AI_CONTEXT_MAX_TOTAL_CHARS", "3000")),
        "max_chars_per_chunk": int(os.getenv("AI_CONTEXT_MAX_CHARS_PER_CHUNK", "700")),
    },
    "refusal": {
        "min_chunks": int(os.getenv("AI_REFUSAL_MIN_CHUNKS", "2")),
        "min_chars": int(os.getenv("AI_REFUSAL_MIN_CHARS", "160")),
        "require_rerank_overlap": os.getenv("AI_REFUSAL_REQUIRE_RERANK_OVERLAP", "1").strip().lower() in {"1", "true", "yes", "on"},
        "min_rerank_score": float(os.getenv("AI_REFUSAL_MIN_RERANK_SCORE", "0.3")),
    },
    "query_rewrite": {
        "enabled": os.getenv("AI_QUERY_REWRITE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"},
        "min_query_length": int(os.getenv("AI_QUERY_REWRITE_MIN_LENGTH", "14")),
        "min_rewrite_length": int(os.getenv("AI_QUERY_REWRITE_MIN_REWRITE_LENGTH", "4")),
        "complex_punctuation_threshold": int(os.getenv("AI_QUERY_REWRITE_PUNCT_THRESHOLD", "2")),
        "fallback_min_results": int(os.getenv("AI_QUERY_REWRITE_FALLBACK_MIN_RESULTS", "1")),
        "trigger_markers": [
            item.strip()
            for item in os.getenv(
                "AI_QUERY_REWRITE_MARKERS",
                "如何,怎么,步骤,流程,排查,定位,分析,总结,介绍,有哪些,以及,并且,区别",
            ).split(",")
            if item.strip()
        ],
    },
    "dense": {
        "candidate_multiplier": int(os.getenv("AI_DENSE_CANDIDATE_MULTIPLIER", "5")),
        "candidate_min": int(os.getenv("AI_DENSE_CANDIDATE_MIN", "15")),
    },
    "hybrid": {
        "dense_k": int(os.getenv("AI_HYBRID_DENSE_K", "7")),
        "keyword_k": int(os.getenv("AI_HYBRID_KEYWORD_K", "7")),
        "final_k": int(os.getenv("AI_HYBRID_FINAL_K", "0")),
        "dense_weight": float(os.getenv("AI_HYBRID_DENSE_WEIGHT", "0.6")),
        "keyword_weight": float(os.getenv("AI_HYBRID_KEYWORD_WEIGHT", "0.4")),
        "keyword_pool_size": int(os.getenv("AI_HYBRID_KEYWORD_POOL", "120"))
    }
}
