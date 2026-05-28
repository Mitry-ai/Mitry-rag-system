# pdf_optimizer.py
import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
    from langchain_community.document_loaders import PDFMinerLoader, PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from retrieval import normalize_metadata
    print("✅ PDF处理库导入成功")
except ImportError as e:
    print(f"❌ PDF处理库导入失败: {e}")

class PDFProcessor:
    """优化的PDF处理器"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.supported_loaders = self._get_supported_loaders()
    
    def _get_supported_loaders(self):
        """获取可用的PDF加载器"""
        loaders = {}
        
        # 按效率排序：PyMuPDF > PDFMiner > PyPDF > Unstructured
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loaders['pymupdf'] = {
                'name': 'PyMuPDF',
                'loader': PyMuPDFLoader,
                'efficiency': 'high',
                'image_handling': 'good'
            }
        except:
            pass
            
        try:
            from langchain_community.document_loaders import PDFMinerLoader
            loaders['pdfminer'] = {
                'name': 'PDFMiner',
                'loader': PDFMinerLoader,
                'efficiency': 'medium',
                'image_handling': 'basic'
            }
        except:
            pass
            
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loaders['pypdf'] = {
                'name': 'PyPDF',
                'loader': PyPDFLoader,
                'efficiency': 'low',
                'image_handling': 'poor'
            }
        except:
            pass
            
        return loaders
    
    def get_best_loader(self):
        """获取最佳可用加载器"""
        for loader_id in ['pymupdf', 'pdfminer', 'pypdf']:
            if loader_id in self.supported_loaders:
                return self.supported_loaders[loader_id]
        return None
    
    def process_pdf_batch(self, file_paths: List[str], progress_callback=None) -> List[Document]:
        """批量处理PDF文件"""
        if not self.supported_loaders:
            raise Exception("没有可用的PDF加载器")
        
        best_loader = self.get_best_loader()
        print(f"🎯 使用PDF加载器: {best_loader['name']}")
        
        all_documents = []
        processed_count = 0
        
        def process_single_pdf(file_path):
            """处理单个PDF文件"""
            try:
                loader = best_loader['loader'](file_path)
                documents = loader.load()
                
                # 为文档添加源文件信息
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(file_path)
                    doc.metadata['file_path'] = file_path
                    doc.metadata['processed_time'] = time.time()
                    doc.metadata = normalize_metadata(doc.metadata)
                
                return documents, file_path, None
            except Exception as e:
                return None, file_path, str(e)
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_pdf, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result, processed_file, error = future.result()
                    if error:
                        print(f"❌ 处理文件失败 {file_path}: {error}")
                    else:
                        all_documents.extend(result)
                        processed_count += 1
                        print(f"✅ 已处理 {processed_count}/{len(file_paths)}: {os.path.basename(file_path)}")
                        
                        if progress_callback:
                            progress_callback(processed_count, len(file_paths))
                            
                except Exception as e:
                    print(f"❌ 处理文件异常 {file_path}: {e}")
        
        return all_documents

    def extract_text_fast(self, file_path: str) -> str:
        """快速提取PDF文本（不处理图片）"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"❌ 快速提取失败 {file_path}: {e}")
            return ""

# 全局PDF处理器实例
pdf_processor = PDFProcessor(max_workers=3)
