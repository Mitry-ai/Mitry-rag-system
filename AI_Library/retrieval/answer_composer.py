from typing import Any, Dict, List, Tuple


def should_refuse(
    context_package: Dict[str, Any],
    min_chunks: int = 2,
    min_chars: int = 160,
    require_rerank_overlap: bool = True,
    min_rerank_score: float = 0.3,
) -> Tuple[bool, str]:
    budget = dict((context_package or {}).get("budget", {}) or {})
    items = list((context_package or {}).get("items", []) or [])
    selected_count = int(budget.get("selected_count", 0) or 0)
    used_chars = int(budget.get("used_chars", 0) or 0)

    if selected_count <= 0:
        return True, "未检索到可用证据。"
    if selected_count < int(min_chunks):
        return True, "证据条目不足，无法形成可靠回答。"
    if used_chars < int(min_chars):
        return True, "证据文本过少，无法形成可靠回答。"

    reranked_items = [item for item in items if item.get("rerank_applied")]
    if require_rerank_overlap and reranked_items:
        max_overlap = 0
        max_score = 0.0
        for item in reranked_items:
            try:
                max_overlap = max(max_overlap, int(item.get("rerank_lexical_overlap") or 0))
            except Exception:
                pass
            try:
                max_score = max(max_score, float(item.get("rerank_lexical_score") or 0.0))
            except Exception:
                pass

        if max_overlap <= 0 or max_score < float(min_rerank_score):
            return True, "检索结果与问题缺少可验证相关性。"

    return False, ""


def build_answer_result(
    query: str,
    status: str,
    answer: str,
    refusal_reason: str,
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query,
        "status": status,
        "answer": answer,
        "refusal_reason": refusal_reason,
        "citations": citations,
        "used_chunks": [str(c.get("chunk_id", "")) for c in citations if c.get("chunk_id")],
    }
