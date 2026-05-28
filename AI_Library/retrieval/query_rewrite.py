import re
from typing import Any, Dict, List


LEADING_PATTERNS = [
    r"^\s*请问",
    r"^\s*请你",
    r"^\s*请帮我",
    r"^\s*帮我",
    r"^\s*麻烦你",
    r"^\s*麻烦",
    r"^\s*我想知道",
    r"^\s*想了解一下",
    r"^\s*想了解",
    r"^\s*能不能",
    r"^\s*能否",
    r"^\s*可以帮我",
]

TRAILING_PATTERNS = [
    r"有哪些[？?]?\s*$",
    r"是什么[？?]?\s*$",
    r"怎么做[？?]?\s*$",
    r"怎么查[？?]?\s*$",
    r"怎么排查[？?]?\s*$",
    r"怎么办[？?]?\s*$",
    r"吗[？?]?\s*$",
    r"呢[？?]?\s*$",
    r"[？?]+\s*$",
]

DEFAULT_TRIGGER_MARKERS = [
    "如何",
    "怎么",
    "步骤",
    "流程",
    "排查",
    "定位",
    "分析",
    "总结",
    "介绍",
    "有哪些",
    "以及",
    "并且",
    "区别",
]


def _to_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_markers(cfg: Dict[str, Any]) -> List[str]:
    markers = cfg.get("trigger_markers", DEFAULT_TRIGGER_MARKERS)
    if isinstance(markers, list):
        return [str(item).strip() for item in markers if str(item).strip()]
    if isinstance(markers, str):
        return [item.strip() for item in markers.split(",") if item.strip()]
    return list(DEFAULT_TRIGGER_MARKERS)


def _normalize_query(text: str) -> str:
    normalized = str(text or "").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _compact_query(text: str) -> str:
    compact = str(text or "")
    compact = compact.replace("（", " ").replace("）", " ")
    compact = compact.replace("(", " ").replace(")", " ")
    compact = re.sub(r"[，,。；;：:、】【\[\]{}<>《》/\\|]+", " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    return compact.strip()


def should_trigger_rewrite(query: str, cfg: Dict[str, Any]) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False

    if not _to_bool(cfg.get("enabled"), True):
        return False

    min_length = _to_int(cfg.get("min_query_length"), 14)
    complex_punctuation = _to_int(cfg.get("complex_punctuation_threshold"), 2)
    markers = _load_markers(cfg)

    punctuation_hits = sum(normalized.count(char) for char in ("，", ",", "、", "；", ";"))
    marker_hit = any(marker in normalized for marker in markers)

    return len(normalized) >= min_length or punctuation_hits >= complex_punctuation or marker_hit


def rewrite_query(query: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    query_raw = _normalize_query(query)
    rewrite_info: Dict[str, Any] = {
        "query_raw": query_raw,
        "query_rewrite": query_raw,
        "effective_query": query_raw,
        "triggered": False,
        "used_rewrite": False,
        "fallback_reason": "",
    }
    if not query_raw:
        rewrite_info["fallback_reason"] = "empty_query"
        return rewrite_info

    if not should_trigger_rewrite(query_raw, cfg):
        rewrite_info["fallback_reason"] = "not_triggered"
        return rewrite_info

    candidate = query_raw
    for pattern in LEADING_PATTERNS:
        candidate = re.sub(pattern, "", candidate)
    for pattern in TRAILING_PATTERNS:
        candidate = re.sub(pattern, "", candidate)

    candidate = candidate.replace("请详细说明", " ")
    candidate = candidate.replace("详细说明", " ")
    candidate = candidate.replace("帮我总结一下", " ")
    candidate = candidate.replace("总结一下", " ")
    candidate = candidate.replace("介绍一下", " ")
    candidate = candidate.replace("解释一下", " ")
    candidate = candidate.replace("帮我看下", " ")
    candidate = candidate.replace("看下", " ")
    candidate = candidate.replace("一下", " ")
    candidate = _compact_query(candidate)

    min_rewrite_length = _to_int(cfg.get("min_rewrite_length"), 4)
    if not candidate:
        rewrite_info["triggered"] = True
        rewrite_info["fallback_reason"] = "empty_rewrite"
        return rewrite_info
    if len(candidate) < min_rewrite_length:
        rewrite_info["triggered"] = True
        rewrite_info["fallback_reason"] = "rewrite_too_short"
        return rewrite_info
    if candidate == query_raw:
        rewrite_info["triggered"] = True
        rewrite_info["fallback_reason"] = "rewrite_same_as_raw"
        return rewrite_info

    rewrite_info["triggered"] = True
    rewrite_info["query_rewrite"] = candidate
    rewrite_info["effective_query"] = candidate
    rewrite_info["used_rewrite"] = True
    return rewrite_info
