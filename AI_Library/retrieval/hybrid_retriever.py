import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    from langchain.schema import BaseRetriever


_CJK_STOPWORDS = {
    "一下",
    "什么",
    "怎么",
    "如何",
    "介绍",
    "详细",
    "哪些",
    "请你",
    "是",
    "的",
    "吗",
    "呢",
}


def _tokenize(text: str) -> List[str]:
    normalized = (text or "").lower()
    for stopword in sorted(_CJK_STOPWORDS, key=len, reverse=True):
        normalized = normalized.replace(stopword, " ")
    tokens: List[str] = []
    for part in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", normalized):
        if re.fullmatch(r"[a-zA-Z0-9]+", part):
            tokens.append(part)
            continue
        if len(part) <= 2:
            tokens.append(part)
            continue
        tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        tokens.append(part)
    return [token for token in tokens if token and token not in _CJK_STOPWORDS]


def _normalize_path(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = str(value).replace("\\", "/")
    return normalized.strip().lower()


def _metadata_values(metadata: Dict[str, Any], keys: List[str]) -> List[str]:
    values: List[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            values.extend([str(v) for v in value if v is not None])
        else:
            values.append(str(value))
    return [v for v in values if v]


def _first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _documents_dir() -> str:
    try:
        from config import DOCUMENTS_DIR
    except ImportError:
        try:
            from ..config import DOCUMENTS_DIR
        except Exception:
            return ""
    return os.path.abspath(DOCUMENTS_DIR)


def _safe_source_path(value: Optional[str]) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""

    path_value = os.path.normpath(raw)
    docs_dir = _documents_dir()
    if os.path.isabs(path_value):
        abs_value = os.path.abspath(path_value)
        if docs_dir:
            try:
                if os.path.commonpath([docs_dir, abs_value]) == docs_dir:
                    return os.path.normpath(os.path.relpath(abs_value, docs_dir))
            except ValueError:
                pass
        return os.path.basename(abs_value)

    parts = [part for part in raw.replace("\\", "/").split("/") if part]
    if parts and parts[0].lower() == "documents":
        parts = parts[1:]
    return os.path.normpath(os.path.join(*parts)) if parts else ""


def _source_filename(value: Optional[str]) -> str:
    safe_path = _safe_source_path(value)
    return os.path.basename(safe_path) if safe_path else ""


def _derive_file_type(source_uri: str, filename: str) -> str:
    suffix = os.path.splitext(filename or source_uri)[1].lower().lstrip(".")
    if suffix:
        return suffix
    return "unknown"


def normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = dict(metadata or {})
    raw_source_uri = _first_non_empty(
        src.get("source_uri"),
        src.get("source"),
        src.get("file_path"),
    )
    safe_source = _safe_source_path(raw_source_uri)
    source_file = _source_filename(_first_non_empty(
        src.get("source_file"),
        src.get("original_filename"),
        safe_source,
    ))
    safe_source = safe_source or source_file
    file_type = _first_non_empty(src.get("file_type"), src.get("type"), _derive_file_type(safe_source, source_file))
    mime_type = _first_non_empty(src.get("mime_type"))

    src["source_uri"] = safe_source
    src["source"] = safe_source
    src["source_file"] = source_file or safe_source
    src["original_filename"] = _source_filename(_first_non_empty(src.get("original_filename"), source_file, safe_source))
    src["file_path"] = safe_source
    src["file_type"] = file_type
    src["type"] = _first_non_empty(src.get("type"), file_type)
    src["mime_type"] = mime_type or src.get("mime_type", "")
    return src


def _to_filter_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    text = str(value).strip()
    return {text.lower()} if text else set()


def _metadata_matches(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
    if not metadata_filter:
        return True

    normalized = normalize_metadata(metadata)
    source_values = {
        _normalize_path(v)
        for v in _metadata_values(normalized, ["source_uri", "source", "file_path"])
        if _normalize_path(v)
    }
    source_file_values = {
        str(v).strip().lower()
        for v in _metadata_values(normalized, ["source_file", "original_filename"])
        if str(v).strip()
    }
    file_type_values = {
        str(v).strip().lower()
        for v in _metadata_values(normalized, ["file_type", "type"])
        if str(v).strip()
    }
    mime_values = {
        str(v).strip().lower()
        for v in _metadata_values(normalized, ["mime_type"])
        if str(v).strip()
    }

    src_filter = _to_filter_set(metadata_filter.get("source_uri") or metadata_filter.get("source"))
    if src_filter:
        normalized_filter = {_normalize_path(v) for v in src_filter if _normalize_path(v)}
        if not normalized_filter.intersection(source_values):
            return False

    src_file_filter = _to_filter_set(metadata_filter.get("source_file") or metadata_filter.get("original_filename"))
    if src_file_filter and not src_file_filter.intersection(source_file_values):
        return False

    file_type_filter = _to_filter_set(metadata_filter.get("file_type") or metadata_filter.get("type"))
    if file_type_filter and not file_type_filter.intersection(file_type_values):
        return False

    mime_filter = _to_filter_set(metadata_filter.get("mime_type"))
    if mime_filter and not mime_filter.intersection(mime_values):
        return False

    return True


def _apply_metadata_filter(docs: List[Document], metadata_filter: Dict[str, Any]) -> List[Document]:
    if not metadata_filter:
        out = []
        for doc in docs:
            doc.metadata = normalize_metadata(doc.metadata)
            out.append(doc)
        return out

    filtered: List[Document] = []
    for doc in docs:
        doc.metadata = normalize_metadata(doc.metadata)
        if _metadata_matches(doc.metadata, metadata_filter):
            filtered.append(doc)
    return filtered


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _document_key(doc: Document) -> str:
    metadata = normalize_metadata(doc.metadata)
    source = metadata.get("source_uri") or metadata.get("source") or metadata.get("source_file") or metadata.get("file_path") or ""
    page = metadata.get("page", "")
    head = (doc.page_content or "")[:120]
    return f"{source}|{page}|{head}"


def _rerank_documents(
    query: str,
    docs: List[Document],
    final_k: int,
    reranker_cfg: Dict[str, Any],
) -> List[Document]:
    if not docs:
        return []

    enabled = _to_bool(reranker_cfg.get("enabled"), default=False)
    if not enabled:
        return docs[:final_k]

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return docs[:final_k]

    candidate_k_default = max(final_k * 3, final_k)
    candidate_k = max(final_k, _to_int(reranker_cfg.get("candidate_k"), candidate_k_default))
    lexical_weight = _to_float(reranker_cfg.get("lexical_weight"), 0.7)
    prior_weight = _to_float(reranker_cfg.get("prior_weight"), 0.3)

    candidates = docs[:candidate_k]
    scored: List[Tuple[float, int, int, float, Document]] = []
    before_keys: List[str] = []

    for rank, doc in enumerate(candidates, start=1):
        text_tokens = set(_tokenize(doc.page_content))
        overlap = len(query_tokens & text_tokens)
        lexical_score = overlap / max(1, len(query_tokens))
        prior_score = 1.0 / rank
        final_score = (lexical_weight * lexical_score) + (prior_weight * prior_score)
        scored.append((final_score, rank, overlap, lexical_score, doc))
        before_keys.append(_document_key(doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    reranked = [doc for _, _, _, _, doc in scored[:final_k]]
    after_keys = [_document_key(doc) for doc in reranked]
    score_map = {
        _document_key(doc): (score, before_rank, overlap, lexical_score)
        for score, before_rank, overlap, lexical_score, doc in scored
    }

    for after_rank, doc in enumerate(reranked, start=1):
        key = _document_key(doc)
        score, before_rank, overlap, lexical_score = score_map.get(key, (0.0, after_rank, 0, 0.0))
        metadata = normalize_metadata(doc.metadata)
        metadata["rerank_applied"] = True
        metadata["rerank_before_rank"] = before_rank
        metadata["rerank_after_rank"] = after_rank
        metadata["rerank_score"] = round(score, 6)
        metadata["rerank_lexical_overlap"] = overlap
        metadata["rerank_lexical_score"] = round(lexical_score, 6)
        metadata["rerank_query_token_count"] = len(query_tokens)
        metadata["rerank_candidate_size"] = len(candidates)
        metadata["rerank_before_order"] = "|".join(before_keys[:10])
        metadata["rerank_after_order"] = "|".join(after_keys[:10])
        doc.metadata = metadata

    return reranked


class FilteredDenseRetriever(BaseRetriever):
    vector_store: Any
    k: int = 3
    candidate_multiplier: int = 5
    candidate_min: int = 15
    metadata_filter: Dict[str, Any] = {}
    reranker: Dict[str, Any] = {}

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Document]:
        try:
            candidate_k = max(self.k, self.k * self.candidate_multiplier, self.candidate_min)
            docs = self.vector_store.similarity_search(query, k=candidate_k)
        except Exception:
            docs = []
        docs = _apply_metadata_filter(docs, self.metadata_filter)
        return _rerank_documents(query=query, docs=docs, final_k=self.k, reranker_cfg=self.reranker)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Document]:
        return self._get_relevant_documents(query=query, run_manager=run_manager)


class HybridRetriever(BaseRetriever):
    vector_store: Any
    dense_k: int = 6
    keyword_k: int = 6
    final_k: int = 3
    dense_weight: float = 0.6
    keyword_weight: float = 0.4
    keyword_pool_size: int = 120
    metadata_filter: Dict[str, Any] = {}
    reranker: Dict[str, Any] = {}

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Document]:
        dense_docs = self._safe_dense_search(query=query, k=self.dense_k)
        keyword_docs = self._keyword_search(query=query, k=self.keyword_k, pool_size=self.keyword_pool_size)
        dense_docs = _apply_metadata_filter(dense_docs, self.metadata_filter)
        keyword_docs = _apply_metadata_filter(keyword_docs, self.metadata_filter)
        fused = self._fuse_ranked_results(dense_docs=dense_docs, keyword_docs=keyword_docs)
        return _rerank_documents(query=query, docs=fused, final_k=self.final_k, reranker_cfg=self.reranker)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Document]:
        return self._get_relevant_documents(query=query, run_manager=run_manager)

    def _safe_dense_search(self, query: str, k: int) -> List[Document]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception:
            return []

    def _keyword_search(self, query: str, k: int, pool_size: int) -> List[Document]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        raw_docs = self._load_keyword_pool(limit=pool_size)
        if not raw_docs:
            return []

        scored_docs: List[Tuple[float, Document]] = []
        for doc in raw_docs:
            text_tokens = set(_tokenize(doc.page_content))
            if not text_tokens:
                continue

            overlap = len(query_tokens & text_tokens)
            if overlap == 0:
                continue

            overlap_ratio = overlap / max(1, len(query_tokens))
            length_penalty = 1.0 / math.log2(2 + max(1, len(text_tokens)))
            score = overlap + (0.5 * overlap_ratio) + (0.2 * length_penalty)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]

    def _load_keyword_pool(self, limit: int) -> List[Document]:
        collection = getattr(self.vector_store, "_collection", None)
        if collection is None:
            return []

        try:
            records = collection.get(include=["documents", "metadatas"], limit=limit)
        except Exception:
            try:
                records = collection.get(include=["documents", "metadatas"])
            except Exception:
                return []

        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []

        out: List[Document] = []
        for idx, text in enumerate(documents):
            if not text:
                continue
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            out.append(Document(page_content=text, metadata=normalize_metadata(metadata)))
        return out

    def _fuse_ranked_results(self, dense_docs: List[Document], keyword_docs: List[Document]) -> List[Document]:
        score_map: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key = _document_key(doc)
            score_map[key] = score_map.get(key, 0.0) + self.dense_weight / rank
            doc_map[key] = doc

        for rank, doc in enumerate(keyword_docs, start=1):
            key = _document_key(doc)
            score_map[key] = score_map.get(key, 0.0) + self.keyword_weight / rank
            doc_map[key] = doc

        ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]


def build_retriever(vector_store: Any, search_k: int, retrieval_config: Dict[str, Any]):
    strategy = str(retrieval_config.get("strategy", "dense")).lower()
    metadata_filter = retrieval_config.get("metadata_filter", {})
    reranker_cfg = retrieval_config.get("reranker", {})
    if strategy == "dense":
        dense_cfg = retrieval_config.get("dense", {})
        return FilteredDenseRetriever(
            vector_store=vector_store,
            k=search_k,
            candidate_multiplier=_to_int(dense_cfg.get("candidate_multiplier"), 5),
            candidate_min=_to_int(dense_cfg.get("candidate_min"), 15),
            metadata_filter=metadata_filter,
            reranker=reranker_cfg,
        )

    hybrid_cfg = retrieval_config.get("hybrid", {})
    return HybridRetriever(
        vector_store=vector_store,
        dense_k=_to_int(hybrid_cfg.get("dense_k"), 7),
        keyword_k=_to_int(hybrid_cfg.get("keyword_k"), 7),
        final_k=_to_int(hybrid_cfg.get("final_k"), search_k),
        dense_weight=float(hybrid_cfg.get("dense_weight", 0.6)),
        keyword_weight=float(hybrid_cfg.get("keyword_weight", 0.4)),
        keyword_pool_size=int(hybrid_cfg.get("keyword_pool_size", 120)),
        metadata_filter=metadata_filter,
        reranker=reranker_cfg,
    )
