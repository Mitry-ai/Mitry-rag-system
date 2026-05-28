import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List


_AUDIT_WRITE_LOCK = threading.Lock()


def audit_enabled(config: Dict[str, Any]) -> bool:
    return bool((config or {}).get("enabled"))


def new_trace_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else default
    except Exception:
        return default


def _snippet(text: Any, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip()


def build_chunk_records(context_package: Dict[str, Any], snippet_chars: int = 140) -> List[Dict[str, Any]]:
    max_chars = _to_int(snippet_chars, 140)
    chunks: List[Dict[str, Any]] = []
    for item in (context_package or {}).get("items", []) or []:
        chunks.append(
            {
                "rank": item.get("rank"),
                "source": str(item.get("source", "")),
                "page": str(item.get("page", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "rerank_score": item.get("rerank_score"),
                "snippet": _snippet(item.get("text", ""), max_chars),
            }
        )
    return chunks


def write_audit_record(config: Dict[str, Any], record: Dict[str, Any]) -> None:
    if not audit_enabled(config):
        return

    path = str((config or {}).get("log_path", "") or "").strip()
    if not path:
        return

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        line = json.dumps(record, ensure_ascii=False, default=str)
        with _AUDIT_WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        print(f"⚠️ 审计日志写入失败: {exc}")
