import os
from typing import Any, Dict, List


def _snippet(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return f"{clean[:max_chars].rstrip()}..."


def _display_source(citation: Dict[str, Any]) -> str:
    for key in ("source_file", "source_uri", "source"):
        value = str(citation.get(key, "") or "").strip()
        if not value:
            continue
        normalized = value.replace("\\", "/").rstrip("/")
        filename = os.path.basename(normalized)
        return filename or normalized
    return "未知文档"


def build_citations(
    context_package: Dict[str, Any],
    max_citations: int = 4,
    snippet_chars: int = 140,
) -> List[Dict[str, Any]]:
    items = list((context_package or {}).get("items", []) or [])
    citations: List[Dict[str, Any]] = []
    for item in items[: max(0, int(max_citations))]:
        citations.append(
            {
                "rank": int(item.get("rank", len(citations) + 1) or (len(citations) + 1)),
                "source": str(item.get("source", "未知文档")),
                "page": str(item.get("page", "未知")),
                "chunk_id": str(item.get("chunk_id", "")),
                "source_uri": str(item.get("source_uri", "")),
                "source_file": str(item.get("source_file", "")),
                "snippet": _snippet(item.get("text", ""), int(snippet_chars)),
            }
        )
    return citations


def render_citations(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return "（无可用引用）"
    lines: List[str] = []
    for idx, citation in enumerate(citations, start=1):
        lines.append(
            f"[{idx}] source={_display_source(citation)} | "
            f"page={citation.get('page', '未知')} | "
            f"chunk_id={citation.get('chunk_id', '')}\n"
            f"    snippet={citation.get('snippet', '')}"
        )
    return "\n".join(lines)
