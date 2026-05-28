from typing import Any, Dict, List


def _to_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _source_label(metadata: Dict[str, Any]) -> str:
    return (
        str(metadata.get("source"))
        or str(metadata.get("source_file"))
        or str(metadata.get("source_uri"))
        or "未知文档"
    )


def _page_label(metadata: Dict[str, Any]) -> str:
    page = metadata.get("page")
    if page is None or str(page).strip() == "":
        return "未知"
    return str(page)


def _chunk_id(metadata: Dict[str, Any], rank: int, source: str, page: str) -> str:
    existing = metadata.get("chunk_id")
    if existing:
        return str(existing)
    return f"{source}:{page}:{rank}"


def build_context_package(query: str, docs: List[Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    max_chunks = _to_int((cfg or {}).get("max_chunks"), 6)
    max_total_chars = _to_int((cfg or {}).get("max_total_chars"), 3000)
    max_chars_per_chunk = _to_int((cfg or {}).get("max_chars_per_chunk"), 700)

    items: List[Dict[str, Any]] = []
    sources: List[str] = []
    used_chars = 0
    truncated = False
    seen_keys = set()

    for rank, doc in enumerate(docs or [], start=1):
        if len(items) >= max_chunks or used_chars >= max_total_chars:
            truncated = True
            break

        metadata = dict(getattr(doc, "metadata", {}) or {})
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue

        source = _source_label(metadata)
        page = _page_label(metadata)

        text = text[:max_chars_per_chunk]
        dedupe_key = f"{source}|{page}|{text[:120]}".lower()
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        remain = max_total_chars - used_chars
        if remain <= 0:
            truncated = True
            break
        if len(text) > remain:
            text = text[:remain]
            truncated = True

        item = {
            "rank": rank,
            "source": source,
            "source_uri": str(metadata.get("source_uri", "")),
            "source_file": str(metadata.get("source_file", "")),
            "page": page,
            "chunk_id": _chunk_id(metadata, rank, source, page),
            "rerank_applied": metadata.get("rerank_applied"),
            "rerank_score": metadata.get("rerank_score"),
            "rerank_lexical_overlap": metadata.get("rerank_lexical_overlap"),
            "rerank_lexical_score": metadata.get("rerank_lexical_score"),
            "rerank_query_token_count": metadata.get("rerank_query_token_count"),
            "text": text,
        }
        items.append(item)
        used_chars += len(text)

        if source not in sources:
            sources.append(source)

    context_blocks: List[str] = []
    for item in items:
        context_blocks.append(
            f"[证据{item['rank']}] 来源: {item['source']} | 页码: {item['page']}\n{item['text']}"
        )

    return {
        "query": query,
        "context_text": "\n\n".join(context_blocks),
        "items": items,
        "sources": sources,
        "budget": {
            "max_chunks": max_chunks,
            "max_total_chars": max_total_chars,
            "max_chars_per_chunk": max_chars_per_chunk,
            "used_chars": used_chars,
            "selected_count": len(items),
            "truncated": truncated,
        },
    }
