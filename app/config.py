import os
import yaml

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_path = os.path.join(_project_root, "config.yml")

_config = {}

def _load():
    global _config
    if _config:
        return
    if os.path.exists(_config_path):
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}

def _resolve(path: str, default=None):
    _load()
    keys = path.split(".")
    node = _config
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
        else:
            return default
        if node is None:
            return default
    return node


# ============================================================
# 便捷访问函数
# ============================================================

def get_llm_base_url() -> str:
    return os.environ.get("CHAT_URL", _resolve("llm.base_url", ""))

def get_llm_model() -> str:
    return os.environ.get("CHAT_MODEL", _resolve("llm.model", ""))

def get_llm_max_tokens() -> int:
    return _resolve("llm.max_tokens", 2048)

def get_llm_temperature() -> float:
    return _resolve("llm.temperature", 0.3)

def get_rewrite_base_url() -> str:
    return os.environ.get("REWRITE_URL", _resolve("llm_rewrite.base_url", get_llm_base_url()))

def get_rewrite_model() -> str:
    return os.environ.get("REWRITE_MODEL", _resolve("llm_rewrite.model", get_llm_model()))

def get_embed_base_url() -> str:
    return os.environ.get("EMBED_URL", _resolve("embedding.base_url", ""))

def get_embed_model() -> str:
    return os.environ.get("EMBED_MODEL", _resolve("embedding.model", ""))

def get_embed_dimension() -> int:
    return _resolve("embedding.dimension", 1024)

def get_rerank_base_url() -> str:
    return os.environ.get("RERANK_URL", _resolve("rerank.base_url", ""))

def get_rerank_model() -> str:
    return os.environ.get("RERANK_MODEL", _resolve("rerank.model", ""))

def get_rerank_api_key() -> str:
    return os.environ.get("SILICONFLOW_API_KEY", _resolve("rerank.api_key", ""))

def get_db_path() -> str:
    path = _resolve("retrieval.db_path", "./data_pipeline/milvus_lite.db")
    if not os.path.isabs(path):
        path = os.path.join(_project_root, path)
    return path

def get_collection_name() -> str:
    return _resolve("retrieval.collection_name", "report_slices")

def get_rag_top_k() -> int:
    return _resolve("retrieval.rag_top_k", 10)

def get_rerank_top_k() -> int:
    return _resolve("retrieval.rerank_top_k", 3)

def get_max_rounds() -> int:
    return _resolve("short_term_memory.max_rounds", 5)

def get_decay_factor() -> float:
    return _resolve("short_term_memory.decay_factor", 0.9)

def get_server_port() -> int:
    return _resolve("server.port", 8000)

def get_server_host() -> str:
    return _resolve("server.host", "0.0.0.0")

def get_metadata_path() -> str:
    path = _resolve("paths.metadata", "./app/data_pipeline/report_template/metadata.json")
    if not os.path.isabs(path):
        path = os.path.join(_project_root, path)
    return path