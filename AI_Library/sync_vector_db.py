#!/usr/bin/env python3
"""同步 documents 文件夹与本地 Chroma 向量库。"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

EXTS = {".pdf", ".txt"}
COMMANDS = {
    "sync-delete": "删除向量库中存在、但 documents 文件夹中已不存在的文件",
    "sync-add": "添加 documents 文件夹中新出现的 PDF/TXT 文件",
    "full-sync": "先执行 sync-delete，再执行 sync-add",
    "rebuild": "重构向量数据库，从 documents 文件夹全量重建",
    "stats": "查看向量库和 documents 文件夹的文件/块统计",
}
HELP = """\
常用命令:
  python sync_vector_db.py stats                查看当前向量数据库统计
  python sync_vector_db.py rebuild              重构向量数据库
  python sync_vector_db.py rebuild --yes        无确认重构向量数据库
  python sync_vector_db.py sync-delete          同步删除已移除文件
  python sync_vector_db.py sync-add             同步新增文件
  python sync_vector_db.py full-sync            同步删除后再同步新增
  python sync_vector_db.py sync-add --dry-run   只查看将新增哪些文件

命令说明:
  sync-delete   {sync_delete}
  sync-add      {sync_add}
  full-sync     {full_sync}
  rebuild       {rebuild}
  stats         {stats}
""".format(
    sync_delete=COMMANDS["sync-delete"],
    sync_add=COMMANDS["sync-add"],
    full_sync=COMMANDS["full-sync"],
    rebuild=COMMANDS["rebuild"],
    stats=COMMANDS["stats"],
)

Chroma = OllamaEmbeddings = TextLoader = None
DATABASE_CONFIG = DOCUMENTS_DIR = SYSTEM_CONFIG = None
build_text_splitter = resolve_chunk_policy = pdf_processor = normalize_metadata = None
chroma_collection_metadata = chroma_persist_directory = None
DEPS_LOADED = False


def no_proxy_for_ollama():
    hosts = ["localhost", "127.0.0.1", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        items = [x.strip() for x in os.environ.get(key, "").split(",") if x.strip()]
        for host in hosts:
            if host not in items:
                items.append(host)
        os.environ[key] = ",".join(items)


def load_deps():
    global Chroma, OllamaEmbeddings, TextLoader
    global DATABASE_CONFIG, DOCUMENTS_DIR, SYSTEM_CONFIG
    global build_text_splitter, resolve_chunk_policy, pdf_processor, normalize_metadata
    global chroma_collection_metadata, chroma_persist_directory
    global DEPS_LOADED
    if DEPS_LOADED:
        return True
    try:
        try:
            from langchain_chroma import Chroma as _Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma as _Chroma
        try:
            from langchain_ollama import OllamaEmbeddings as _OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings as _OllamaEmbeddings
        from langchain_community.document_loaders import TextLoader as _TextLoader
        from chunk_policy import build_text_splitter as _build_splitter
        from chunk_policy import resolve_chunk_policy as _resolve_policy
        from config import DATABASE_CONFIG as _DB_CONFIG
        from config import DOCUMENTS_DIR as _DOCS_DIR
        from config import SYSTEM_CONFIG as _SYS_CONFIG
        from config import chroma_collection_metadata as _chroma_collection_metadata
        from config import chroma_persist_directory as _chroma_persist_directory
        from pdf_optimizer import pdf_processor as _pdf_processor
        from retrieval import normalize_metadata as _normalize_metadata
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    Chroma, OllamaEmbeddings, TextLoader = _Chroma, _OllamaEmbeddings, _TextLoader
    DATABASE_CONFIG, DOCUMENTS_DIR, SYSTEM_CONFIG = _DB_CONFIG, _DOCS_DIR, _SYS_CONFIG
    build_text_splitter, resolve_chunk_policy = _build_splitter, _resolve_policy
    pdf_processor, normalize_metadata = _pdf_processor, _normalize_metadata
    chroma_collection_metadata = _chroma_collection_metadata
    chroma_persist_directory = _chroma_persist_directory
    DEPS_LOADED = True
    print("✅ 依赖库导入成功")
    return True


def confirm(prompt, yes=False):
    return yes or input(prompt).strip().lower() == "y"


def chroma_index_error(error):
    msg = str(error).lower()
    return "cannot open header file" in msg or ("header" in msg and "hnsw" in msg)


class Syncer:
    def __init__(self):
        if not load_deps():
            raise RuntimeError("依赖库加载失败")
        self.docs_dir = Path(DOCUMENTS_DIR)
        self.db_dir = Path(DATABASE_CONFIG["vector_db_path"])
        self.embeddings = None
        self.db = None

    def key(self, path):
        if not path:
            return ""
        raw = os.path.normpath(str(path).strip())
        if not raw:
            return ""
        if os.path.isabs(raw):
            try:
                return os.path.normpath(os.path.relpath(raw, str(self.docs_dir)))
            except ValueError:
                return os.path.basename(raw)
        parts = [p for p in raw.replace("\\", os.sep).split(os.sep) if p]
        if parts and parts[0].lower() == self.docs_dir.name.lower():
            parts = parts[1:]
        return os.path.normpath(os.path.join(*parts)) if parts else ""

    def init_embeddings(self):
        if self.embeddings is not None:
            return True
        try:
            base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.embeddings = OllamaEmbeddings(
                        model=SYSTEM_CONFIG["embedding_model"],
                        base_url=base_url,
                    )
                except TypeError:
                    self.embeddings = OllamaEmbeddings(model=SYSTEM_CONFIG["embedding_model"])
            return True
        except Exception as e:
            print(f"❌ Embedding 初始化失败: {e}")
            return False

    def check_embeddings(self):
        try:
            if not self.embeddings.embed_query("health check"):
                raise RuntimeError("Ollama 返回了空 embedding")
            return True
        except Exception as e:
            print("❌ Ollama embedding 服务不可用，已停止操作。")
            print(f"   使用模型: {SYSTEM_CONFIG['embedding_model']}")
            print(f"   Ollama 地址: {os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")
            print(f"   如模型不存在: ollama pull {SYSTEM_CONFIG['embedding_model']}")
            print("   如果使用代理，请确保 NO_PROXY 包含 localhost,127.0.0.1,::1。")
            print(f"   详细错误: {e}")
            return False

    def chroma(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Chroma(
                persist_directory=chroma_persist_directory(path),
                embedding_function=self.embeddings,
                collection_metadata=chroma_collection_metadata(),
            )

    def open_db(self):
        if self.db is not None:
            return True
        if not self.init_embeddings():
            return False
        try:
            self.db = self.chroma(self.db_dir)
            print("✅ 加载现有向量数据库" if self.db_dir.exists() else "ℹ️  向量数据库不存在，将创建新数据库")
            return True
        except Exception as e:
            print(f"❌ 初始化向量数据库失败: {e}")
            return False

    def persist(self, db=None):
        db = db or self.db
        if db is None or not hasattr(db, "persist"):
            return
        try:
            import chromadb

            major, minor = [int(x) for x in chromadb.__version__.split(".")[:2]]
            if (major, minor) >= (0, 4):
                return
        except Exception:
            pass
        db.persist()

    def close_chroma(self, db, persist=True):
        if db is None:
            return
        if persist:
            try:
                self.persist(db)
            except Exception:
                pass
        try:
            manager = getattr(getattr(getattr(db, "_client", None), "_server", None), "_manager", None)
            self.close_vector_segments(manager)
        except Exception:
            pass
        try:
            system = getattr(getattr(db, "_client", None), "_system", None)
            if system is not None and hasattr(system, "stop"):
                system.stop()
        except Exception:
            pass
        gc.collect()
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    def close_vector_segments(self, manager):
        if manager is None:
            return
        seen = set()
        instances = list(getattr(manager, "_instances", {}).values())
        file_cache = getattr(manager, "_vector_instances_file_handle_cache", None)
        cache_values = list(getattr(getattr(file_cache, "cache", {}), "values", lambda: [])())
        for instance in instances + cache_values:
            marker = id(instance)
            if marker in seen:
                continue
            seen.add(marker)
            if not hasattr(instance, "close_persistent_index"):
                continue
            self.flush_vector_segment(instance)
            try:
                instance.close_persistent_index()
            except Exception:
                pass
        cache = getattr(file_cache, "cache", None)
        if cache is not None:
            try:
                cache.clear()
            except Exception:
                pass

    def flush_vector_segment(self, instance):
        try:
            curr_batch = getattr(instance, "_curr_batch", None)
            if curr_batch is not None and len(curr_batch) > 0 and hasattr(instance, "_apply_batch"):
                instance._apply_batch(curr_batch)
                try:
                    from chromadb.segment.impl.vector.batch import Batch

                    instance._curr_batch = Batch()
                except Exception:
                    pass
                brute_force = getattr(instance, "_brute_force_index", None)
                if brute_force is not None and hasattr(brute_force, "clear"):
                    brute_force.clear()
            if hasattr(instance, "_persist"):
                instance._persist()
        except Exception:
            pass

    def close(self, persist=True):
        db, self.db = self.db, None
        self.close_chroma(db, persist=persist)

    def folder_files(self):
        if not self.docs_dir.exists():
            print(f"❌ documents 文件夹不存在: {self.docs_dir}")
            return {}
        files = {}
        for path in self.docs_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTS:
                files[self.key(path)] = {"full_path": str(path), "size": path.stat().st_size}
        return files

    def vector_files(self):
        if not self.open_db():
            return {}
        try:
            records = self.db._collection.get(include=["metadatas"])
        except Exception as e:
            print(f"❌ 获取向量数据库文件失败: {e}")
            return {}
        files = {}
        for doc_id, metadata in zip(records.get("ids") or [], records.get("metadatas") or []):
            metadata = normalize_metadata(metadata or {})
            source = metadata.get("source_uri") or metadata.get("source") or metadata.get("source_file")
            key = self.key(source)
            if key:
                files.setdefault(key, {"ids": [], "count": 0})
                files[key]["ids"].append(doc_id)
                files[key]["count"] += 1
        return files

    def counts(self, vector, folder):
        print(f"📊 向量数据库中有 {len(vector)} 个文件")
        print(f"📁 documents文件夹中有 {len(folder)} 个文件")

    def load_docs(self, paths):
        pdfs = [p for p in paths if Path(p).suffix.lower() == ".pdf"]
        txts = [p for p in paths if Path(p).suffix.lower() == ".txt"]
        docs, loaded, failed = [], [], []
        print(f"📄 处理 {len(pdfs)} 个PDF和 {len(txts)} 个TXT文件")
        if pdfs:
            try:
                pdf_docs = pdf_processor.process_pdf_batch(pdfs)
                for doc in pdf_docs:
                    doc.metadata = normalize_metadata(doc.metadata)
                docs.extend(pdf_docs)
                loaded.extend(pdfs)
            except Exception as e:
                print(f"❌ PDF处理失败: {e}")
                failed.extend(pdfs)
        for path in txts:
            try:
                text_docs = TextLoader(path, encoding="utf-8").load()
                for doc in text_docs:
                    doc.metadata["source_file"] = os.path.basename(path)
                    doc.metadata["file_path"] = path
                    doc.metadata = normalize_metadata(doc.metadata)
                docs.extend(text_docs)
                loaded.append(path)
            except Exception as e:
                print(f"❌ 处理TXT文件失败 {path}: {e}")
                failed.append(path)
        if failed:
            print(f"⚠️  以下文件处理失败: {', '.join(os.path.basename(p) for p in failed)}")
        return docs, loaded

    def split(self, docs):
        print(f"✂️ 分割 {len(docs)} 个文档...")
        policy = resolve_chunk_policy()
        print(f"📐 切块策略: {policy['name']} (size={policy['chunk_size']}, overlap={policy['chunk_overlap']})")
        chunks = build_text_splitter(policy["name"]).split_documents(docs)
        print(f"✅ 分割为 {len(chunks)} 个文本块")
        return chunks

    def add_batches(self, db, chunks):
        batch_size = int(os.getenv("AI_CHROMA_ADD_BATCH_SIZE", "32"))
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            db.add_documents(chunks[start:end])
            print(f"   已写入向量块 {end}/{len(chunks)}")
        self.persist(db)

    def ensure_queryable(self, db):
        # Chroma 0.4.x 有时只写 index_metadata.pickle。构建后查询一次可强制生成 HNSW 主索引文件。
        try:
            db.similarity_search("health check", k=1)
        except Exception as e:
            raise RuntimeError(f"向量库写入后无法查询: {e}") from e

    def build_db(self, target, paths):
        if target.exists():
            shutil.rmtree(target)
        docs, loaded = self.load_docs(paths)
        if not docs:
            raise RuntimeError("没有成功加载任何文档")
        chunks = self.split(docs)
        db = None
        try:
            print("🗃️ 创建向量数据库...")
            db = self.chroma(target)
            self.add_batches(db, chunks)
            self.ensure_queryable(db)
        finally:
            self.close_chroma(db, persist=True)
        return len(loaded), len(chunks)

    def replace_db(self, temp):
        backup = self.db_dir.with_name(f"{self.db_dir.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
        if backup.exists():
            shutil.rmtree(backup)

    def exec_replace_db(self, temp, file_count, chunk_count):
        code = r"""
import os
import json
import shutil
import sys
import time

with open(sys.argv[1], "r", encoding="utf-8") as f:
    params = json.load(f)

temp = params["temp"]
db_dir = params["db_dir"]
file_count = params["file_count"]
chunk_count = params["chunk_count"]
backup = db_dir + "_backup_" + time.strftime("%Y%m%d_%H%M%S")

try:
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(backup, exist_ok=True)
    moved_old = []
    for name in os.listdir(db_dir):
        src = os.path.join(db_dir, name)
        dst = os.path.join(backup, name)
        shutil.move(src, dst)
        moved_old.append((dst, src))
    for name in os.listdir(temp):
        shutil.move(os.path.join(temp, name), os.path.join(db_dir, name))
    shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(temp, ignore_errors=True)
    try:
        os.remove(sys.argv[1])
        os.remove(sys.argv[0])
    except Exception:
        pass
    print(f"✅ 向量数据库重建成功: {file_count} 个文件，{chunk_count} 个文本块")
except Exception as e:
    for old_path, restore_path in reversed(locals().get("moved_old", [])):
        if os.path.exists(old_path) and not os.path.exists(restore_path):
            shutil.move(old_path, restore_path)
    print(f"❌ 替换向量数据库失败: {e}")
    sys.exit(1)
"""
        helper_path = Path(tempfile.gettempdir()) / f"ai_vector_replace_{os.getpid()}.py"
        params_path = Path(tempfile.gettempdir()) / f"ai_vector_replace_{os.getpid()}.json"
        helper_path.write_text(code, encoding="utf-8")
        params_path.write_text(
            json.dumps(
                {
                    "temp": str(temp),
                    "db_dir": str(self.db_dir),
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.execv(
            sys.executable,
            [sys.executable, str(helper_path), str(params_path)],
        )
        backup.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        moved_old = []
        try:
            for child in list(self.db_dir.iterdir()):
                target = backup / child.name
                shutil.move(str(child), str(target))
                moved_old.append((target, self.db_dir / child.name))

            for child in list(temp.iterdir()):
                shutil.move(str(child), str(self.db_dir / child.name))
        except Exception:
            for old_path, restore_path in reversed(moved_old):
                if old_path.exists() and not restore_path.exists():
                    shutil.move(str(old_path), str(restore_path))
            raise
        if backup.exists():
            shutil.rmtree(backup)

    def rebuild_current(self):
        folder = self.folder_files()
        if not folder:
            print("❌ documents文件夹中没有 PDF/TXT 文件")
            return False
        if not self.init_embeddings() or not self.check_embeddings():
            return False
        self.close(persist=False)
        temp = self.db_dir.with_name(f"{self.db_dir.name}_rebuild_{os.getpid()}_{int(time.time())}")
        try:
            paths = [item["full_path"] for item in folder.values()]
            print(f"📄 开始处理 {len(paths)} 个文件...")
            file_count, chunk_count = self.build_db(temp, paths)
            print("🔁 临时向量库已验证，正在释放 Chroma 句柄并替换旧库...")
            sys.stdout.flush()
            sys.stderr.flush()
            self.exec_replace_db(temp, file_count, chunk_count)
            return True
        except PermissionError as e:
            print("❌ 重建失败: 向量数据库文件正在被占用")
            print("   请关闭 Web 服务、其它 Python 进程或数据库查看器后重试。")
            print(f"   详细错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 重建失败: {e}")
            return False
        finally:
            if temp.exists():
                try:
                    shutil.rmtree(temp)
                except Exception:
                    pass

    def rebuild(self, yes=False):
        print("🔄 开始重构向量数据库...")
        if not confirm("⚠️  这将删除现有向量数据库并重新构建，确认？(y/N): ", yes):
            print("❌ 操作取消")
            return False
        return self.rebuild_current()

    def sync_delete(self, dry_run=False, yes=False):
        print("🔄 开始同步删除的文件...")
        vector, folder = self.vector_files(), self.folder_files()
        self.counts(vector, folder)
        stale = [key for key in vector if key not in folder and key != "sample"]
        if not stale:
            print("✅ 没有需要删除的文件")
            return True
        print(f"🗑️ 发现 {len(stale)} 个需要删除的文件:")
        for key in stale:
            print(f"   - {key} ({vector[key]['count']} 个文档块)")
        if dry_run:
            print("🔍 干运行模式，不会实际删除")
            return True
        if not confirm("⚠️  确认删除这些文件？(y/N): ", yes):
            print("❌ 操作取消")
            return False
        errors = []
        for key in stale:
            try:
                self.db._collection.delete(ids=vector[key]["ids"])
                print(f"✅ 已删除: {key}")
            except Exception as e:
                errors.append(e)
                print(f"❌ 删除失败 {key}: {e}")
        if errors and any(chroma_index_error(e) for e in errors):
            print("⚠️  检测到 Chroma 索引文件损坏，将通过重建向量库完成同步删除...")
            return self.rebuild_current()
        if errors:
            print(f"❌ 同步删除失败，{len(errors)} 个文件未删除")
            return False
        self.persist()
        print(f"✅ 同步完成，删除了 {len(stale)} 个文件")
        return True

    def sync_add(self, dry_run=False, yes=False):
        print("🔄 开始同步新增的文件...")
        vector, folder = self.vector_files(), self.folder_files()
        self.counts(vector, folder)
        new_paths = [folder[key]["full_path"] for key in folder if key not in vector]
        if not new_paths:
            print("✅ 没有需要添加的新文件")
            return True
        print(f"📥 发现 {len(new_paths)} 个需要添加的文件:")
        for path in new_paths:
            print(f"   - {os.path.basename(path)}")
        if dry_run:
            print("🔍 干运行模式，不会实际添加")
            return True
        if not confirm("⚠️  确认添加这些文件？(y/N): ", yes):
            print("❌ 操作取消")
            return False
        try:
            docs, loaded = self.load_docs(new_paths)
            if docs:
                self.add_batches(self.db, self.split(docs))
                self.ensure_queryable(self.db)
            print(f"✅ 同步完成，成功添加了 {len(loaded)} 个文件")
            return True
        except Exception as e:
            if chroma_index_error(e):
                print("⚠️  检测到 Chroma 索引文件损坏，将通过重建向量库完成同步新增...")
                return self.rebuild_current()
            print(f"❌ 同步新增失败: {e}")
            return False

    def full_sync(self, dry_run=False, yes=False):
        print("🔄 开始完全同步...")
        if not self.sync_delete(dry_run=dry_run, yes=yes):
            return False
        if not self.sync_add(dry_run=dry_run, yes=yes):
            return False
        print("✅ 完全同步完成")
        return True

    def stats(self):
        print("📊 向量数据库统计信息:")
        vector, folder = self.vector_files(), self.folder_files()
        print(f"  文档文件数: {len(vector)}")
        print(f"  文档块总数: {sum(x['count'] for x in vector.values())}")
        print(f"  文件夹文件数: {len(folder)}")
        only_vector = sorted(set(vector) - set(folder) - {"sample"})
        only_folder = sorted(set(folder) - set(vector))
        if only_vector:
            print(f"  🗑️  仅在向量库中: {len(only_vector)} 个文件")
            for key in only_vector[:5]:
                print(f"     - {key}")
        if only_folder:
            print(f"  📁  仅在文件夹中: {len(only_folder)} 个文件")
            for key in only_folder[:5]:
                print(f"     - {key}")


def parser():
    p = argparse.ArgumentParser(
        description="向量数据库同步工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP,
    )
    p.add_argument("action", choices=list(COMMANDS), help="操作类型，可用 -h 查看完整说明")
    p.add_argument("--dry-run", action="store_true", help="只显示将执行的同步操作，不写入数据库")
    p.add_argument("--yes", action="store_true", help="跳过确认提示")
    return p


def main():
    no_proxy_for_ollama()
    args = parser().parse_args()
    try:
        syncer = Syncer()
    except RuntimeError:
        sys.exit(1)
    actions = {
        "sync-delete": lambda: syncer.sync_delete(args.dry_run, args.yes),
        "sync-add": lambda: syncer.sync_add(args.dry_run, args.yes),
        "full-sync": lambda: syncer.full_sync(args.dry_run, args.yes),
        "rebuild": lambda: syncer.rebuild(args.yes),
        "stats": lambda: syncer.stats() is None,
    }
    success = actions[args.action]()
    syncer.close(persist=True)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
