# ai_library.py (完整修改版)
import os
import sys
import threading
import time
import shutil
import html
from urllib.parse import quote
from typing import Optional, Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
os.chdir(CURRENT_DIR)

def _configure_console_stream(stream):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


_configure_console_stream(sys.stdout)
_configure_console_stream(sys.stderr)

print("🚀 启动纯本地AI智库系统...")

try:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    import gradio as gr
    from network import InternetEnabledAI
    from auth import UserManager
    from config import (
        AI_MODES,
        ADMIN_SECONDARY_PASSWORD,
        AUDIT_CONFIG,   
        DATABASE_CONFIG,
        DOCUMENTS_DIR,
        RETRIEVAL_CONFIG,
        SYSTEM_CONFIG,
        chroma_collection_metadata,
        chroma_persist_directory,
    )
    from pdf_optimizer import pdf_processor  # 新增
    from chunk_policy import build_text_splitter, resolve_chunk_policy
    from audit import build_chunk_records, new_trace_id, now_iso, write_audit_record
    from retrieval import (
        build_answer_result,
        build_citations,
        build_context_package,
        build_retriever,
        normalize_metadata,
        render_citations,
        rewrite_query,
        should_refuse,
    )
    print("✅ 核心包导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class AILibrarySystem:
    """AI智库系统主类"""
    
    def __init__(self):
        self.user_manager = UserManager(DATABASE_CONFIG["db_path"])
        self.current_users = {}  # {session_id: user_info}
        self.max_concurrent_questions = max(
            1, int(SYSTEM_CONFIG.get("max_concurrent_users", 1) or 1)
        )
        self.question_semaphore = threading.BoundedSemaphore(self.max_concurrent_questions)
        self.active_questions = 0
        self.active_question_lock = threading.Lock()
        self.session_lock = threading.RLock()
        self.vector_db_lock = threading.RLock()
        self.ingest_lock = threading.Lock()
        self.vector_db = None
        self.llm = None
        self.net_ai = None
        self.qa_chains = {}  # 不同模式的QA链
        self.doc_count = 0

    def _register_session_user(self, session_id: str, user: Dict[str, Any]):
        with self.session_lock:
            self.current_users[session_id] = dict(user)

    def _get_session_user(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        with self.session_lock:
            user = self.current_users.get(session_id)
            return dict(user) if isinstance(user, dict) else user

    def _remove_session_user(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        with self.session_lock:
            return self.current_users.pop(session_id, None)

    def _acquire_question_slot(self):
        self.question_semaphore.acquire()
        with self.active_question_lock:
            self.active_questions += 1

    def _release_question_slot(self):
        with self.active_question_lock:
            if self.active_questions > 0:
                self.active_questions -= 1
        self.question_semaphore.release()
        
    def initialize_system(self, force_reload: bool = False):
        """初始化系统组件"""
        print("🔧 初始化系统组件...")
        
        # 检查是否已有向量数据库
        if not force_reload and os.path.exists(DATABASE_CONFIG["vector_db_path"]):
            print("📂 加载现有向量数据库...")
            try:
                embeddings = OllamaEmbeddings(model=SYSTEM_CONFIG["embedding_model"])
                with self.vector_db_lock:
                    self.vector_db = Chroma(
                        persist_directory=chroma_persist_directory(),
                        embedding_function=embeddings,
                        collection_metadata=chroma_collection_metadata(),
                    )
                    self.doc_count = self.vector_db._collection.count()
                print(f"✅ 加载现有向量数据库，包含 {self.doc_count} 个文档")
            except Exception as e:
                print(f"❌ 加载向量数据库失败: {e}")
                force_reload = True
        
        if force_reload or self.vector_db is None:
            self._create_vector_database()
        
        # 初始化语言模型
        print("🤖 初始化语言模型...")
        try:
            self.llm = Ollama(model=SYSTEM_CONFIG["default_model"], temperature=0.1)
            test_response = self.llm.invoke("你好")
            print(f"✅ 语言模型就绪: {test_response[:20]}...")
        except Exception as e:
            print(f"❌ 语言模型失败: {e}")
            return False
        
        # 初始化网络AI
        print("🌐 初始化网络AI...")
        try:
            self.net_ai = InternetEnabledAI(model_name=SYSTEM_CONFIG["default_model"])
            print("✅ 网络AI就绪")
        except Exception as e:
            print(f"❌ 网络AI初始化失败: {e}")
            return False
        
        # 为每种模式创建QA链
        self._create_qa_chains()
        
        return True
    
    def _create_vector_database(self):
        """创建向量数据库"""
        with self.vector_db_lock:
            print("📚 加载文档...")
            documents = []
            
            if not os.path.exists(DOCUMENTS_DIR):
                os.makedirs(DOCUMENTS_DIR)
                self._create_sample_documents()
            
            try:
                pdf_loader = DirectoryLoader(DOCUMENTS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
                pdf_docs = pdf_loader.load()
                for doc in pdf_docs:
                    doc.metadata = normalize_metadata(doc.metadata)
                documents.extend(pdf_docs)
            except:
                pass
            
            try:
                txt_loader = DirectoryLoader(DOCUMENTS_DIR, glob="**/*.txt", loader_cls=TextLoader)
                txt_docs = txt_loader.load()
                for doc in txt_docs:
                    doc.metadata = normalize_metadata(doc.metadata)
                documents.extend(txt_docs)
            except:
                pass
            
            if len(documents) == 0:
                from langchain.schema import Document
                documents = [Document(page_content="这是示例文档内容。", metadata=normalize_metadata({"source": "sample", "file_type": "txt"}))]
            
            print(f"✅ 已加载 {len(documents)} 个文档")
            
            # 文本分割
            print("✂️ 分割文本...")
            policy = resolve_chunk_policy()
            print(
                f"📐 切块策略: {policy['name']} "
                f"(size={policy['chunk_size']}, overlap={policy['chunk_overlap']})"
            )
            text_splitter = build_text_splitter(policy["name"])
            text_chunks = text_splitter.split_documents(documents)
            print(f"生成 {len(text_chunks)} 个文本片段")
            
            # 创建向量数据库
            print("🗃️ 创建向量数据库...")
            try:
                embeddings = OllamaEmbeddings(model=SYSTEM_CONFIG["embedding_model"])
                self.vector_db = Chroma.from_documents(
                    documents=text_chunks,
                    embedding=embeddings,
                    persist_directory=chroma_persist_directory(),
                    collection_metadata=chroma_collection_metadata(),
                )
                self.doc_count = len(documents)
                print("✅ 向量数据库创建成功")
            except Exception as e:
                print(f"❌ 向量数据库失败: {e}")
                raise
    
    def _create_sample_documents(self):
        """创建示例文档"""
        sample_content = [
            "人工智能(AI)是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。",
            "机器学习是人工智能的一种应用，为系统提供了在无需明确编程的情况下学习和改进的能力。",
            "深度学习是机器学习的一个子集，使用具有多个层的神经网络来模拟人脑的复杂模式。",
            "RAG（检索增强生成）架构结合了信息检索和大语言模型，提高回答的准确性和可靠性。"
        ]
        for i, content in enumerate(sample_content):
            sample_path = os.path.join(DOCUMENTS_DIR, f"sample_{i+1}.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(content)
        print("📝 已创建示例文档")
    
    def _create_qa_chains(self):
        """为不同模式创建本地问答编排组件"""
        with self.vector_db_lock:
            print("🔗 创建多模式QA链...")
            qa_chains = {}
            for mode_name, config in AI_MODES.items():
                # 为每种模式创建特定配置的LLM
                mode_llm = Ollama(
                    model=SYSTEM_CONFIG["default_model"],
                    temperature=config.temperature
                )
                
                # 构建可切换的检索器（dense/hybrid）
                retriever = build_retriever(
                    vector_store=self.vector_db,
                    search_k=config.search_k,
                    retrieval_config=RETRIEVAL_CONFIG
                )
                
                qa_chains[mode_name] = {
                    "llm": mode_llm,
                    "retriever": retriever,
                }
                strategy = RETRIEVAL_CONFIG.get("strategy", "dense")
                print(f"✅ {config.name}模式QA链创建完成 (retrieval={strategy})")
            self.qa_chains = qa_chains

    def set_metadata_filter(self, metadata_filter: Dict[str, Any]):
        """在代码层动态更新检索 metadata 过滤条件。"""
        with self.vector_db_lock:
            RETRIEVAL_CONFIG["metadata_filter"] = metadata_filter or {}
        self._create_qa_chains()
    
    def add_documents(self, file_paths: List[str]) -> tuple:
        """添加多个文档到向量数据库（优化版）"""
        try:
            with self.ingest_lock:
                documents = []
                successful_files = []
                failed_files = []
                
                # 分离PDF和TXT文件
                pdf_files = [f for f in file_paths if f.endswith('.pdf')]
                txt_files = [f for f in file_paths if f.endswith('.txt')]
                
                print(f"📄 开始处理 {len(pdf_files)} 个PDF和 {len(txt_files)} 个TXT文件")
                
                # 处理PDF文件（使用优化的处理器）
                if pdf_files:
                    print("🔄 使用优化处理器处理PDF文件...")
                    try:
                        pdf_documents = pdf_processor.process_pdf_batch(pdf_files)
                        documents.extend(pdf_documents)
                        successful_files.extend([os.path.basename(f) for f in pdf_files])
                        print(f"✅ 成功处理 {len(pdf_files)} 个PDF文件")
                    except Exception as e:
                        print(f"❌ PDF处理失败: {e}")
                        failed_files.extend([os.path.basename(f) for f in pdf_files])
                
                # 处理TXT文件
                if txt_files:
                    print("📝 处理TXT文件...")
                    for txt_file in txt_files:
                        try:
                            loader = TextLoader(txt_file, encoding='utf-8')
                            txt_docs = loader.load()
                            for doc in txt_docs:
                                doc.metadata['source_file'] = os.path.basename(txt_file)
                                doc.metadata['file_path'] = txt_file
                                doc.metadata = normalize_metadata(doc.metadata)
                            documents.extend(txt_docs)
                            successful_files.append(os.path.basename(txt_file))
                        except Exception as e:
                            print(f"❌ 处理TXT文件失败 {txt_file}: {e}")
                            failed_files.append(os.path.basename(txt_file))
                
                if documents:
                    print(f"✂️ 开始分割 {len(documents)} 个文档...")
                    policy = resolve_chunk_policy()
                    print(
                        f"📐 切块策略: {policy['name']} "
                        f"(size={policy['chunk_size']}, overlap={policy['chunk_overlap']})"
                    )
                    text_splitter = build_text_splitter(policy["name"])
                    text_chunks = text_splitter.split_documents(documents)
                    print(f"✅ 分割为 {len(text_chunks)} 个文本块")
                    
                    # 添加到现有向量数据库
                    print("🗃️ 更新向量数据库...")
                    with self.vector_db_lock:
                        if self.vector_db is None:
                            embeddings = OllamaEmbeddings(model=SYSTEM_CONFIG["embedding_model"])
                            self.vector_db = Chroma.from_documents(
                                documents=text_chunks,
                                embedding=embeddings,
                                persist_directory=chroma_persist_directory(),
                                collection_metadata=chroma_collection_metadata(),
                            )
                        else:
                            self.vector_db.add_documents(text_chunks)
                            self.vector_db.persist()
                        self.doc_count = self.vector_db._collection.count()
                    
                    # 重新创建QA链
                    print("🔗 更新QA链...")
                    self._create_qa_chains()
                    
                    print(f"✅ 成功添加 {len(successful_files)} 个文件到知识库")
                    if failed_files:
                        print(f"⚠️  {len(failed_files)} 个文件处理失败")
                    return True, successful_files, failed_files
                else:
                    print("❌ 没有成功加载任何文档")
                    return False, [], file_paths
                 
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False, [], file_paths

    def _document_key(self, path: str) -> str:
        """把文档路径规整成相对 documents 目录的稳定 key。"""
        if not path:
            return ""

        raw = os.path.normpath(str(path).strip())
        if not raw:
            return ""

        docs_root = os.path.abspath(DOCUMENTS_DIR)
        if os.path.isabs(raw):
            try:
                relative = os.path.relpath(os.path.abspath(raw), docs_root)
            except ValueError:
                return os.path.basename(raw)
        else:
            parts = [part for part in raw.replace("\\", os.sep).split(os.sep) if part]
            if parts and parts[0].lower() == os.path.basename(docs_root).lower():
                parts = parts[1:]
            relative = os.path.join(*parts) if parts else ""

        if not relative or relative == ".":
            return ""
        return os.path.normpath(relative)

    def _document_key_aliases(self, *paths: str) -> List[str]:
        """生成文档匹配别名，兼容历史绝对路径和脱敏后的文件名。"""
        aliases: List[str] = []
        for path in paths:
            key = self._document_key(path)
            for candidate in (key, os.path.basename(key) if key else ""):
                if candidate and candidate not in aliases:
                    aliases.append(candidate)
        return aliases

    def _vector_document_index(self) -> Dict[str, Dict[str, Any]]:
        """返回向量库里按文档 key 聚合的 chunk id 和数量。"""
        index: Dict[str, Dict[str, Any]] = {}
        with self.vector_db_lock:
            collection = getattr(self.vector_db, "_collection", None)
            if collection is None:
                return index

            try:
                records = collection.get(include=["metadatas"])
            except Exception as e:
                print(f"❌ 读取向量库文档索引失败: {e}")
                return index

        for doc_id, raw_metadata in zip(records.get("ids") or [], records.get("metadatas") or []):
            raw_metadata = raw_metadata or {}
            metadata = normalize_metadata(raw_metadata)
            aliases = self._document_key_aliases(
                raw_metadata.get("source_uri"),
                raw_metadata.get("source"),
                raw_metadata.get("file_path"),
                raw_metadata.get("source_file"),
                metadata.get("source_uri"),
                metadata.get("source"),
                metadata.get("file_path"),
                metadata.get("source_file"),
            )
            for key in aliases:
                item = index.setdefault(key, {"ids": [], "count": 0, "_ids_seen": set()})
                if doc_id in item["_ids_seen"]:
                    continue
                item["_ids_seen"].add(doc_id)
                item["ids"].append(doc_id)
                item["count"] = len(item["ids"])
        return index

    def _vector_ids_for_document_key(self, vector_index: Dict[str, Dict[str, Any]], key: str) -> List[str]:
        exact_key = self._document_key(key)
        if exact_key and vector_index.get(exact_key, {}).get("ids"):
            return list(dict.fromkeys(vector_index[exact_key]["ids"]))

        ids: List[str] = []
        seen = set()
        for alias in self._document_key_aliases(key):
            if alias == exact_key:
                continue
            for doc_id in vector_index.get(alias, {}).get("ids", []):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                ids.append(doc_id)
        return ids

    def list_document_files(self) -> List[Dict[str, Any]]:
        """列出 documents 目录中的可管理文档。"""
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        vector_index = self._vector_document_index()
        files: List[Dict[str, Any]] = []
        for root, _, names in os.walk(DOCUMENTS_DIR):
            for name in names:
                if os.path.splitext(name)[1].lower() not in {".pdf", ".txt"}:
                    continue
                full_path = os.path.join(root, name)
                key = self._document_key(full_path)
                try:
                    stat = os.stat(full_path)
                except OSError:
                    continue
                files.append(
                    {
                        "key": key,
                        "name": name,
                        "path": full_path,
                        "type": os.path.splitext(name)[1].lower().lstrip("."),
                        "size": stat.st_size,
                        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                        "chunk_count": len(self._vector_ids_for_document_key(vector_index, key)),
                    }
                )
        files.sort(key=lambda item: item["key"].lower())
        return files

    def delete_documents(self, selected_keys: List[str]) -> Dict[str, Any]:
        """删除 documents 文件，并同步删除向量库 chunks 后刷新问答链。"""
        raw_keys = [selected_keys] if isinstance(selected_keys, str) else list(selected_keys or [])
        selected = [self._document_key(key) for key in raw_keys]
        selected = list(dict.fromkeys(key for key in selected if key))
        if not selected:
            return {"deleted": [], "missing": [], "errors": ["未选择要删除的文档"], "vector_chunks_deleted": 0}

        with self.ingest_lock:
            files_by_key = {item["key"]: item for item in self.list_document_files()}
            vector_index = self._vector_document_index()
            deleted: List[str] = []
            missing: List[str] = []
            errors: List[str] = []
            ids_to_delete: List[str] = []
            vector_sync_failed = False

            for key in selected:
                item = files_by_key.get(key)
                if not item:
                    missing.append(key)
                    continue

                try:
                    os.remove(item["path"])
                    deleted.append(key)
                    ids_to_delete.extend(self._vector_ids_for_document_key(vector_index, key))
                    print(f"🗑️ 已删除文档文件: {key}")
                except Exception as e:
                    errors.append(f"{key}: {e}")

            if deleted:
                ids_to_delete = list(dict.fromkeys(ids_to_delete))
                try:
                    with self.vector_db_lock:
                        if ids_to_delete and self.vector_db is not None:
                            self.vector_db._collection.delete(ids=ids_to_delete)
                            if hasattr(self.vector_db, "persist"):
                                self.vector_db.persist()
                        if self.vector_db is not None:
                            self.doc_count = self.vector_db._collection.count()
                    print(f"✅ 已同步删除 {len(ids_to_delete)} 个向量块")
                except Exception as e:
                    vector_sync_failed = True
                    errors.append(f"向量库同步删除失败: {e}")

                if not vector_sync_failed:
                    print("🔗 删除后更新QA链...")
                    self._create_qa_chains()

            return {
                "deleted": deleted,
                "missing": missing,
                "errors": errors,
                "vector_chunks_deleted": len(ids_to_delete),
                "vector_sync_failed": vector_sync_failed,
                "qa_chain_updated": bool(deleted) and not vector_sync_failed,
            }

    def _retrieve_documents(self, retriever: Any, query: str) -> List[Any]:
        if hasattr(retriever, "invoke"):
            documents = retriever.invoke(query)
        else:
            documents = retriever.get_relevant_documents(query)

        if not isinstance(documents, list):
            return [documents] if documents else []
        return documents

    def _format_rewrite_note(self, rewrite_info: Dict[str, Any]) -> str:
        query_raw = str(rewrite_info.get("query_raw", "") or "")
        query_rewrite = str(rewrite_info.get("query_rewrite", "") or "")
        effective_query = str(rewrite_info.get("effective_query", query_raw) or query_raw)
        fallback_reason = str(rewrite_info.get("fallback_reason", "") or "")
        if not query_raw:
            return ""
        if query_rewrite and query_rewrite != query_raw and effective_query == query_rewrite:
            return (
                f"🔎 检索改写:\n原始: {query_raw}\n检索: {effective_query}\n"
                f"状态: 已启用 rewrite\n\n"
            )
        if query_rewrite and query_rewrite != query_raw and effective_query == query_raw:
            return (
                f"🔎 检索改写:\n原始: {query_raw}\n候选改写: {query_rewrite}\n检索: {effective_query}\n"
                f"状态: 已回退 ({fallback_reason or 'rewrite_fallback'})\n\n"
            )
        return f"🔎 检索改写:\n原始: {query_raw}\n检索: {effective_query}\n状态: 未触发\n\n"

    def ask_question(self, question: str, mode: str = "balanced", session_id: str = None) -> str:
        """处理用户问题"""
        if not question.strip():
            return "请输入有效问题"
        
        trace_id = new_trace_id()
        start_time = time.perf_counter()
        network_enabled = False
        retrieval_strategy = ""
        self._acquire_question_slot()
        try:
            # 检测是否需要网络功能
            function_name, _ = self.net_ai.detect_function_call(question)
            
            if function_name:
                # 使用网络AI处理
                network_enabled = True
                retrieval_strategy = "tool"
                result = self.net_ai.chat(question)
                user = self._get_session_user(session_id)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                audit_record = {
                    "trace_id": trace_id,
                    "timestamp": now_iso(),
                    "session_id": session_id or "",
                    "user_id": str(user.get("id", "")) if user else "",
                    "mode": mode,
                    "retrieval_strategy": retrieval_strategy,
                    "network_enabled": True,
                    "question": question,
                    "query_raw": question,
                    "query_rewrite": "",
                    "effective_query": question,
                    "fallback_reason": "",
                    "rerank_enabled": False,
                    "context_budget": {
                        "selected_count": 0,
                        "used_chars": 0,
                        "max_chunks": 0,
                    },
                    "chunks": [],
                    "tool_name": function_name,
                    "status": "answered",
                    "refusal_reason": "",
                    "latency_ms": latency_ms,
                }
                if AUDIT_CONFIG.get("include_answer"):
                    audit_record["answer"] = result
                write_audit_record(AUDIT_CONFIG, audit_record)
                return result
            else:
                # 使用本地RAG处理（retriever -> context builder -> llm）
                retrieval_strategy = RETRIEVAL_CONFIG.get("strategy", "")
                rewrite_cfg = RETRIEVAL_CONFIG.get("query_rewrite", {})
                rewrite_info = rewrite_query(question, rewrite_cfg)
                retrieval_query = rewrite_info.get("effective_query", question) or question
                fallback_min_results = int(rewrite_cfg.get("fallback_min_results", 1) or 1)

                with self.vector_db_lock:
                    qa_pipeline = self.qa_chains.get(mode, self.qa_chains["balanced"])
                    retriever = qa_pipeline["retriever"]
                    mode_llm = qa_pipeline["llm"]
                    source_documents = self._retrieve_documents(retriever, retrieval_query)

                    if (
                        rewrite_info.get("used_rewrite")
                        and (not source_documents or len(source_documents) < fallback_min_results)
                    ):
                    #如果有效证据过少，用原始问题rebank
                        rewrite_info["effective_query"] = question
                        rewrite_info["used_rewrite"] = False
                        rewrite_info["fallback_reason"] = "rewrite_low_results"
                        source_documents = self._retrieve_documents(retriever, question)

                context_cfg = RETRIEVAL_CONFIG.get("context_builder", {})
                context_package = build_context_package(
                    query=rewrite_info.get("effective_query", question),
                    docs=source_documents,
                    cfg=context_cfg,
                )
                context_text = context_package.get("context_text", "")
                citations = build_citations(context_package)
                refusal_cfg = RETRIEVAL_CONFIG.get("refusal", {})
                should_refuse_answer, refusal_reason = should_refuse(
                    context_package,
                    min_chunks=int(refusal_cfg.get("min_chunks", 2) or 2),
                    min_chars=int(refusal_cfg.get("min_chars", 160) or 160),
                    require_rerank_overlap=bool(refusal_cfg.get("require_rerank_overlap", True)),
                    min_rerank_score=float(refusal_cfg.get("min_rerank_score", 0.3) or 0.3),
                )

                if should_refuse_answer:
                    answer_result = build_answer_result(
                        query=question,
                        status="refused",
                        answer="抱歉，当前证据不足，无法给出可靠回答。",
                        refusal_reason=refusal_reason,
                        citations=citations,
                    )
                else:
                    prompt = (
                        "你是企业内部知识库问答助手。请优先依据给定证据回答。"
                        "如果证据不足，明确说明信息不足，不要编造。\n\n"
                        f"问题:\n{question}\n\n"
                        f"证据上下文:\n{context_text}"
                    )
                    llm_output = mode_llm.invoke(prompt)
                    answer = llm_output if isinstance(llm_output, str) else str(llm_output)
                    answer_result = build_answer_result(
                        query=question,
                        status="answered",
                        answer=answer,
                        refusal_reason="",
                        citations=citations,
                    )

                mode_name = AI_MODES.get(mode, AI_MODES["balanced"]).name
                citation_text = render_citations(answer_result.get("citations", []))
                rewrite_note = self._format_rewrite_note(rewrite_info)
                user = self._get_session_user(session_id)
                budget = dict(context_package.get("budget", {}) or {})
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                audit_record = {
                    "trace_id": trace_id,
                    "timestamp": now_iso(),
                    "session_id": session_id or "",
                    "user_id": str(user.get("id", "")) if user else "",
                    "mode": mode,
                    "retrieval_strategy": retrieval_strategy,
                    "network_enabled": False,
                    "question": question,
                    "query_raw": rewrite_info.get("query_raw", question),
                    "query_rewrite": rewrite_info.get("query_rewrite", ""),
                    "effective_query": rewrite_info.get("effective_query", question),
                    "fallback_reason": rewrite_info.get("fallback_reason", ""),
                    "rerank_enabled": bool(RETRIEVAL_CONFIG.get("reranker", {}).get("enabled")),
                    "context_budget": {
                        "selected_count": budget.get("selected_count", 0),
                        "used_chars": budget.get("used_chars", 0),
                        "max_chunks": budget.get("max_chunks", 0),
                    },
                    "chunks": build_chunk_records(
                        context_package,
                        snippet_chars=AUDIT_CONFIG.get("snippet_chars", 140),
                    ),
                    "status": answer_result.get("status", ""),
                    "refusal_reason": answer_result.get("refusal_reason", ""),
                    "latency_ms": latency_ms,
                }
                if AUDIT_CONFIG.get("include_answer"):
                    audit_record["answer"] = answer_result.get("answer", "")
                write_audit_record(AUDIT_CONFIG, audit_record)

                if answer_result.get("status") == "refused":
                    return (
                        f"🎯 模式: {mode_name}\n\n"
                        f"{rewrite_note}"
                        f"⚠️ 已触发拒答\n"
                        f"refusal_reason: {answer_result.get('refusal_reason', '')}\n\n"
                        f"{answer_result.get('answer', '')}\n\n"
                        f"📚 结构化引用:\n{citation_text}"
                    )

                return (
                    f"🎯 模式: {mode_name}\n\n"
                    f"{rewrite_note}"
                    f"{answer_result.get('answer', '')}\n\n"
                    f"📚 结构化引用:\n{citation_text}"
                )
                
        except Exception as e:
            user = self._get_session_user(session_id)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            write_audit_record(
                AUDIT_CONFIG,
                {
                    "trace_id": trace_id,
                    "timestamp": now_iso(),
                    "session_id": session_id or "",
                    "user_id": str(user.get("id", "")) if user else "",
                    "mode": mode,
                    "retrieval_strategy": retrieval_strategy,
                    "network_enabled": network_enabled,
                    "question": question,
                    "query_raw": question,
                    "query_rewrite": "",
                    "effective_query": question,
                    "fallback_reason": "",
                    "rerank_enabled": bool(RETRIEVAL_CONFIG.get("reranker", {}).get("enabled")) if not network_enabled else False,
                    "context_budget": {
                        "selected_count": 0,
                        "used_chars": 0,
                        "max_chunks": 0,
                    },
                    "chunks": [],
                    "status": "error",
                    "refusal_reason": "",
                    "latency_ms": latency_ms,
                    "error": str(e),
                },
            )
            return f"❌ 处理问题时出错: {str(e)}"
        finally:
            self._release_question_slot()

# 全局系统实例
system = AILibrarySystem()

def main():
    """主函数"""
    # 初始化系统
    if not system.initialize_system():
        print("❌ 系统初始化失败")
        return
    
    print("✅ RAG系统就绪!")
    print(
        f"🧾 审计日志: {'开启' if AUDIT_CONFIG.get('enabled') else '关闭'} | "
        f"path={AUDIT_CONFIG.get('log_path')}"
    )
    write_audit_record(
        AUDIT_CONFIG,
        {
            "trace_id": new_trace_id(),
            "timestamp": now_iso(),
            "event": "startup",
            "status": "startup",
            "audit_enabled": bool(AUDIT_CONFIG.get("enabled")),
            "log_path": AUDIT_CONFIG.get("log_path", ""),
        },
    )
    
    def login(username, password):
        """用户登录"""
        user = system.user_manager.authenticate_user(username, password)
        if user:
            session_id = system.user_manager.create_session(user['id'])
            system._register_session_user(session_id, user)
            return (
                f"✅ 登录成功！欢迎 {user['username']}",
                session_id,
                gr.update(visible=user['role'] == 'admin'),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        return (
            "❌ 登录失败，请检查用户名和密码",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    
    def logout(session_id):
        """用户登出"""
        if session_id:
            system.user_manager.delete_session(session_id)
            system._remove_session_user(session_id)
        return "已登出", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def _get_active_user(session_id):
        """从会话表刷新用户信息，避免权限变更后继续使用旧角色。"""
        if not session_id:
            return None
        user = system.user_manager.validate_session(session_id)
        if user:
            system._register_session_user(session_id, user)
            return user
        system._remove_session_user(session_id)
        return None

    def _require_admin(session_id, action="执行该操作"):
        user = _get_active_user(session_id)
        if not user:
            return None, "🔐 请先登录"
        if user.get("role") != "admin":
            return None, f"❌ 只有管理员可以{action}"
        return user, ""
    
    def _as_file_list(files):
        if not files:
            return []
        if isinstance(files, (list, tuple)):
            return [file for file in files if file]
        return [files]

    def _uploaded_file_path(file):
        if isinstance(file, dict):
            return file.get("path") or file.get("name") or ""
        return str(getattr(file, "name", file) or "")

    def _uploaded_file_name(file):
        if isinstance(file, dict):
            return os.path.basename(file.get("orig_name") or file.get("name") or file.get("path") or "")
        return os.path.basename(getattr(file, "orig_name", "") or _uploaded_file_path(file))

    def _build_upload_queue(files, current_queue):
        """多次选择文件时累积待上传列表。"""
        queue = list(current_queue or [])
        seen_paths = {item.get("path") for item in queue if item.get("path")}

        for file in _as_file_list(files):
            file_path = _uploaded_file_path(file)
            if not file_path or file_path in seen_paths:
                continue
            queue.append({
                "path": file_path,
                "name": _uploaded_file_name(file),
            })
            seen_paths.add(file_path)

        if not queue:
            return queue, "请选择要上传的文件"

        status = f"已选择 {len(queue)} 个待上传文件:\n"
        for item in queue:
            status += f"• {item.get('name') or os.path.basename(item.get('path', ''))}\n"
        return queue, status

    def upload_files(files, session_id, progress=gr.Progress()):
        """多文件上传处理（带进度显示）"""
        user = _get_active_user(session_id)
        if not user:
            return "❌ 请先登录", files

        if user['role'] != 'admin':
            return "❌ 只有管理员可以上传文件", files
        
        if not files:
            return "请选择要上传的文件", files
        
        try:
            # 保存文件到documents目录
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            saved_paths = []
            file_details = []
            
            progress(0, desc="开始文件上传...")
            
            for i, file in enumerate(files):
                # 获取文件名
                source_path = _uploaded_file_path(file)
                filename = _uploaded_file_name(file) or os.path.basename(source_path)
                save_path = os.path.join(DOCUMENTS_DIR, filename)
                
                # 如果文件已存在，添加序号
                counter = 1
                original_save_path = save_path
                while os.path.exists(save_path):
                    name, ext = os.path.splitext(filename)
                    save_path = os.path.join(DOCUMENTS_DIR, f"{name}_{counter}{ext}")
                    counter += 1
                
                # 复制文件到目标目录
                shutil.copy2(source_path, save_path)
                saved_paths.append(save_path)
                file_details.append(os.path.basename(save_path))
                
                progress((i + 1) / len(files), desc=f"上传文件中... ({i + 1}/{len(files)})")
            
            # 添加到向量数据库
            progress(0, desc="处理文件中...")
            success, processed_files, failed_files = system.add_documents(saved_paths)
            
            if success:
                success_msg = f"✅ 成功上传 {len(processed_files)} 个文件并更新知识库:\n"
                for f in processed_files:
                    success_msg += f"• {f}\n"
                
                if failed_files:
                    success_msg += f"\n⚠️  以下文件处理失败:\n"
                    for f in failed_files:
                        success_msg += f"• {f}\n"
                 
                return success_msg, []
            else:
                return f"❌ 文件处理失败，成功处理了 {len(processed_files)} 个文件", []
                 
        except Exception as e:
            return f"❌ 上传失败: {str(e)}", files

    def _format_size(num_bytes):
        try:
            size = float(num_bytes)
        except Exception:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024
        return f"{size:.1f} GB"

    def _document_download_href(path):
        normalized = os.path.abspath(path).replace("\\", "/")
        return "/file=" + quote(normalized, safe="/")

    def _format_document_table(documents):
        if not documents:
            return "<p>当前 documents 文件夹没有 PDF/TXT 文档。</p>"

        rows = []
        for doc in documents:
            filename = html.escape(doc["key"])
            href = html.escape(_document_download_href(doc["path"]), quote=True)
            download_name = html.escape(doc["name"], quote=True)
            rows.append(
                "<tr>"
                f'<td><a href="{href}" download="{download_name}">{filename}</a></td>'
                f"<td>{html.escape(doc['type'].upper())}</td>"
                f"<td style=\"text-align:right\">{html.escape(_format_size(doc['size']))}</td>"
                f"<td>{html.escape(doc['modified'])}</td>"
                f"<td style=\"text-align:right\">{int(doc['chunk_count'])}</td>"
                "</tr>"
            )
        return (
            "<table>"
            "<thead><tr>"
            "<th>文档</th><th>类型</th><th>大小</th><th>修改时间</th><th>向量块</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    def _document_delete_options(documents):
        options = []
        for doc in documents:
            label = (
                f"{doc['key']} | {doc['type'].upper()} | "
                f"{_format_size(doc['size'])} | {doc['chunk_count']} 个向量块"
            )
            options.append((label, doc["key"]))
        return options

    def refresh_admin_documents(session_id, status=""):
        user, error = _require_admin(session_id, "管理文档")
        if not user:
            return (
                "<p>请先登录后查看文档。</p>",
                gr.update(choices=[], value=[]),
                error,
            )

        documents = system.list_document_files()
        return (
            _format_document_table(documents),
            gr.update(choices=_document_delete_options(documents), value=[]),
            status or f"共 {len(documents)} 个可管理文档",
        )

    def delete_selected_documents(selected_keys, session_id, progress=gr.Progress()):
        user, error = _require_admin(session_id, "删除文档")
        if not user:
            return refresh_admin_documents(session_id, error)

        selected = (
            list(selected_keys)
            if isinstance(selected_keys, (list, tuple))
            else ([selected_keys] if selected_keys else [])
        )
        if not selected:
            return refresh_admin_documents(session_id, "请选择要删除的文档")

        progress(0, desc="删除文件并同步向量库...")
        result = system.delete_documents(selected)
        progress(1, desc="文档列表已刷新")

        messages = []
        if result.get("deleted"):
            messages.append(f"✅ 已删除 {len(result['deleted'])} 个文档")
            if result.get("vector_sync_failed"):
                messages.append("⚠️ 向量库同步删除未完成")
            else:
                messages.append(f"✅ 已同步删除 {result.get('vector_chunks_deleted', 0)} 个向量块")
            if result.get("qa_chain_updated"):
                messages.append("✅ 已重新更新问答链")
        if result.get("missing"):
            messages.append("⚠️ 以下文档不存在或已被删除: " + ", ".join(result["missing"]))
        if result.get("errors"):
            messages.append("❌ 删除过程中出现错误:\n" + "\n".join(result["errors"]))
        if not messages:
            messages.append("没有执行删除操作")
        return refresh_admin_documents(session_id, "\n".join(messages))

    def _role_label(role):
        return "管理员" if role == "admin" else "普通用户"

    def _format_user_table(users):
        if not users:
            return "<p>当前没有用户。</p>"

        rows = []
        for user in users:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(user.get('username', '')))}</td>"
                f"<td>{html.escape(_role_label(user.get('role')))}</td>"
                f"<td>{html.escape(str(user.get('created_at') or ''))}</td>"
                f"<td>{html.escape(str(user.get('last_login') or '从未登录'))}</td>"
                "</tr>"
            )
        return (
            "<table>"
            "<thead><tr><th>用户名</th><th>权限类型</th><th>创建时间</th><th>最后登录</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    def _user_options(users):
        return [
            (f"{user.get('username', '')} | {_role_label(user.get('role'))}", user.get("username", ""))
            for user in users
            if user.get("username")
        ]

    def refresh_user_management(session_id, status=""):
        user, error = _require_admin(session_id, "管理用户")
        if not user:
            return (
                "<p>当前用户没有用户管理权限。</p>",
                gr.update(choices=[], value=None),
                gr.update(value="user"),
                error,
            )

        users = system.user_manager.list_users()
        return (
            _format_user_table(users),
            gr.update(choices=_user_options(users), value=None),
            gr.update(value="user"),
            status or f"共 {len(users)} 个用户",
        )

    def add_managed_user(username, password, role, session_id):
        user, error = _require_admin(session_id, "添加用户")
        if not user:
            return refresh_user_management(session_id, error)
        success, message = system.user_manager.add_user(username, password, role)
        return refresh_user_management(session_id, ("✅ " if success else "❌ ") + message)

    def delete_managed_user(username, secondary_password, session_id):
        user, error = _require_admin(session_id, "删除用户")
        if not user:
            return refresh_user_management(session_id, error)
        success, message = system.user_manager.delete_user_by_username(
            username,
            current_username=user.get("username"),
            secondary_password=secondary_password,
            admin_secondary_password=ADMIN_SECONDARY_PASSWORD,
        )
        return refresh_user_management(session_id, ("✅ " if success else "❌ ") + message)

    def update_managed_user_role(username, role, session_id):
        user, error = _require_admin(session_id, "修改用户权限")
        if not user:
            return refresh_user_management(session_id, error)
        success, message = system.user_manager.update_user_role(
            username,
            role,
            current_username=user.get("username"),
        )
        return refresh_user_management(session_id, ("✅ " if success else "❌ ") + message)

    def update_managed_user_password(username, new_password, secondary_password, session_id):
        user, error = _require_admin(session_id, "修改用户密码")
        if not user:
            return refresh_user_management(session_id, error)
        success, message = system.user_manager.update_user_password(
            username,
            new_password,
            secondary_password=secondary_password,
            admin_secondary_password=ADMIN_SECONDARY_PASSWORD,
        )
        return refresh_user_management(session_id, ("✅ " if success else "❌ ") + message)

    def show_document_management(session_id):
        user, error = _require_admin(session_id, "管理文档")
        if not user:
            docs_html, doc_choices, doc_status = refresh_admin_documents(session_id, error)
            return gr.update(visible=False), gr.update(visible=False), docs_html, doc_choices, doc_status
        docs_html, doc_choices, doc_status = refresh_admin_documents(session_id)
        return gr.update(visible=True), gr.update(visible=False), docs_html, doc_choices, doc_status

    def show_user_management(session_id):
        user, error = _require_admin(session_id, "管理用户")
        if not user:
            users_html, user_choices, role_update, user_status = refresh_user_management(session_id, error)
            return gr.update(visible=False), gr.update(visible=False), users_html, user_choices, role_update, user_status
        users_html, user_choices, role_update, user_status = refresh_user_management(session_id)
        return gr.update(visible=False), gr.update(visible=True), users_html, user_choices, role_update, user_status

    def ask_question_wrapper(question, mode, session_id):
        """包装的问题处理函数"""
        if not system._get_session_user(session_id):
            return "🔐 请先登录系统"
        return system.ask_question(question, mode, session_id)
    
    # 创建界面
    print("🖥️ 启动用户界面...")
    with gr.Blocks(title="AI智库", theme=gr.themes.Soft()) as demo:
        session_state = gr.State("")  # 存储session_id
        upload_queue_state = gr.State([])  # 累积多次选择的待上传文件
        
        gr.Markdown("# 🧠 AI智库")
        gr.Markdown("支持多模式检索、用户管理和文件上传的智能系统")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 登录区域
                gr.Markdown("### 🔐 用户登录")
                with gr.Group():
                    login_username = gr.Textbox(label="用户名", placeholder="输入用户名")
                    login_password = gr.Textbox(label="密码", placeholder="输入密码", type="password")
                    with gr.Row():
                        login_btn = gr.Button("登录", variant="primary")
                        logout_btn = gr.Button("登出")
                
                login_status = gr.Textbox(label="登录状态", interactive=False)
                
                # 文件上传区域（仅管理员可见）
                upload_section = gr.Group(visible=False)
                with upload_section:
                    gr.Markdown("### 管理员功能")
                    with gr.Row():
                        admin_docs_entry_btn = gr.Button("📚 文档管理", variant="primary")
                        admin_users_entry_btn = gr.Button("👤 用户管理", variant="primary")

                    document_management_panel = gr.Group(visible=False)
                    with document_management_panel:
                        gr.Markdown("### 📁 文档上传（管理员）")
                        gr.Markdown("支持拖入多个文件，也支持多次点击选择文件，确认列表后再上传")
                        file_upload = gr.File(
                            label="拖入文件或一次选择多个文档",
                            file_types=[".pdf", ".txt"],
                            file_count="multiple",
                            height=100
                        )
                        add_file_btn = gr.UploadButton(
                            "➕ 继续选择文件",
                            file_types=[".pdf", ".txt"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("📤 上传文件", variant="primary")
                        upload_status = gr.Textbox(
                            label="上传状态",
                            interactive=False,
                            lines=5,
                            max_lines=10
                        )

                        gr.Markdown("### 📚 文档查看与删除（管理员）")
                        with gr.Row():
                            refresh_docs_btn = gr.Button("🔄 刷新文档")
                            delete_docs_btn = gr.Button("🗑️ 删除选中文档", variant="stop")
                        document_list = gr.HTML()
                        document_delete_select = gr.CheckboxGroup(
                            label="选择要删除的文档",
                            choices=[],
                            value=[],
                        )
                        document_status = gr.Textbox(
                            label="文档管理状态",
                            interactive=False,
                            lines=3,
                            max_lines=6,
                        )

                    user_management_panel = gr.Group(visible=False)
                    with user_management_panel:
                        gr.Markdown("### 👤 用户管理（管理员）")
                        refresh_users_btn = gr.Button("🔄 刷新用户")
                        user_list = gr.HTML()
                        user_status = gr.Textbox(
                            label="用户管理状态",
                            interactive=False,
                            lines=3,
                            max_lines=6,
                        )
                        gr.Markdown("#### 添加用户")
                        add_username = gr.Textbox(label="用户名", placeholder="输入用户名")
                        add_password = gr.Textbox(label="密码", placeholder="输入密码", type="password")
                        add_role = gr.Radio(
                            choices=[("普通用户", "user"), ("管理员", "admin")],
                            label="用户权限类型",
                            value="user",
                        )
                        add_user_btn = gr.Button("➕ 添加用户", variant="primary")
                        gr.Markdown("#### 修改 / 删除用户")
                        user_select = gr.Dropdown(label="选择用户", choices=[], value=None)
                        update_role = gr.Radio(
                            choices=[("普通用户", "user"), ("管理员", "admin")],
                            label="修改权限为",
                            value="user",
                        )
                        with gr.Row():
                            update_role_btn = gr.Button("🔁 修改权限")
                            delete_user_btn = gr.Button("🗑️ 删除用户", variant="stop")
                        reset_password = gr.Textbox(
                            label="新密码",
                            placeholder="普通用户可直接修改；管理员账号需填写二级密码",
                            type="password",
                        )
                        secondary_password = gr.Textbox(
                            label="二级密码",
                            placeholder="修改管理员密码或删除管理员账号时必填",
                            type="password",
                        )
                        reset_password_btn = gr.Button("🔑 修改密码")
            
            with gr.Column(scale=2):
                # 问答区域
                gr.Markdown("### 💬 智能问答")
                
                # AI模式选择
                mode_radio = gr.Radio(
                    choices=[(config.name, mode) for mode, config in AI_MODES.items()],
                    label="选择AI模式",
                    value="balanced",
                    info="选择不同的回答风格和检索策略"
                )
                
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="问关于您文档或网络信息的问题...",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 提问", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空")
                
                answer_output = gr.Textbox(
                    label="AI回答",
                    lines=8,
                    interactive=False
                )
        
        # 系统信息
        gr.Markdown("### 📊 系统信息")
        info_display = gr.Markdown()
        
        def update_system_info():
            """更新系统信息显示"""
            with system.active_question_lock:
                active_questions = system.active_questions
            return f"""
        - **文档数量**: {system.doc_count} 个
        - **语言模型**: {SYSTEM_CONFIG['default_model']} 🟢
        - **嵌入模型**: {SYSTEM_CONFIG['embedding_model']} 🟢
        - **运行模式**: 本地文档 + 联网搜索 + 多模式检索
        - **问答并发上限**: {system.max_concurrent_questions}
        - **当前活跃问答**: {active_questions}
        - **支持格式**: PDF, TXT
            """
        
        # 事件处理
        login_btn.click(
            login,
            inputs=[login_username, login_password],
            outputs=[
                login_status,
                session_state,
                upload_section,
                document_management_panel,
                user_management_panel,
            ]
        ).then(
            update_system_info,
            outputs=info_display
        ).then(
            refresh_admin_documents,
            inputs=[session_state],
            outputs=[document_list, document_delete_select, document_status]
        )
        
        logout_btn.click(
            logout,
            inputs=[session_state],
            outputs=[
                login_status,
                session_state,
                upload_section,
                document_management_panel,
                user_management_panel,
            ]
        ).then(
            update_system_info,
            outputs=info_display
        ).then(
            refresh_admin_documents,
            inputs=[session_state],
            outputs=[document_list, document_delete_select, document_status]
        )

        admin_docs_entry_btn.click(
            show_document_management,
            inputs=[session_state],
            outputs=[
                document_management_panel,
                user_management_panel,
                document_list,
                document_delete_select,
                document_status,
            ]
        )

        admin_users_entry_btn.click(
            show_user_management,
            inputs=[session_state],
            outputs=[
                document_management_panel,
                user_management_panel,
                user_list,
                user_select,
                update_role,
                user_status,
            ]
        )

        file_upload.change(
            _build_upload_queue,
            inputs=[file_upload, upload_queue_state],
            outputs=[upload_queue_state, upload_status]
        )

        add_file_btn.upload(
            _build_upload_queue,
            inputs=[add_file_btn, upload_queue_state],
            outputs=[upload_queue_state, upload_status]
        )
        
        upload_btn.click(
            upload_files,
            inputs=[upload_queue_state, session_state],
            outputs=[upload_status, upload_queue_state]
        ).then(
            update_system_info,
            outputs=info_display
        ).then(
            refresh_admin_documents,
            inputs=[session_state],
            outputs=[document_list, document_delete_select, document_status]
        )

        refresh_docs_btn.click(
            refresh_admin_documents,
            inputs=[session_state],
            outputs=[document_list, document_delete_select, document_status]
        )

        delete_docs_btn.click(
            delete_selected_documents,
            inputs=[document_delete_select, session_state],
            outputs=[document_list, document_delete_select, document_status]
        ).then(
            update_system_info,
            outputs=info_display
        )

        refresh_users_btn.click(
            refresh_user_management,
            inputs=[session_state],
            outputs=[user_list, user_select, update_role, user_status]
        )

        add_user_btn.click(
            add_managed_user,
            inputs=[add_username, add_password, add_role, session_state],
            outputs=[user_list, user_select, update_role, user_status]
        )

        update_role_btn.click(
            update_managed_user_role,
            inputs=[user_select, update_role, session_state],
            outputs=[user_list, user_select, update_role, user_status]
        )

        delete_user_btn.click(
            delete_managed_user,
            inputs=[user_select, secondary_password, session_state],
            outputs=[user_list, user_select, update_role, user_status]
        )

        reset_password_btn.click(
            update_managed_user_password,
            inputs=[user_select, reset_password, secondary_password, session_state],
            outputs=[user_list, user_select, update_role, user_status]
        )
        
        def clear_all():
            return "", ""
        
        submit_btn.click(
            ask_question_wrapper,
            inputs=[question_input, mode_radio, session_state],
            outputs=answer_output
        )
        
        question_input.submit(
            ask_question_wrapper,
            inputs=[question_input, mode_radio, session_state],
            outputs=answer_output
        )
        
        clear_btn.click(
            clear_all,
            outputs=[question_input, answer_output]
        )
        
        # 初始化系统信息显示
        demo.load(update_system_info, outputs=info_display)
    
    server_port = int(os.getenv("AI_WEB_PORT", "7860"))
    print(f"🌐 服务启动在: http://localhost:{server_port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        allowed_paths=[DOCUMENTS_DIR],
    )

if __name__ == "__main__":
    main()
