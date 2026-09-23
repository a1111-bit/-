"""
基于FAISS的医疗RAG引擎（不依赖缺失的 EvoAgentX RAG 模块）

改造说明：
- 原 EvoAgentX 的 rag 模块在本项目副本中缺失（rag/schema.py、rag/readers/ 等不存在）
- 直接使用 FAISS + 智谱 embedding-2 重写
- 对外接口与原版保持一致，下游 workflow.py 无需改动
"""

import os
import json
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import faiss
import numpy as np
import PyPDF2
from zhipuai import ZhipuAI

from evoagentx_medical_config import (
    DATA_DIR, CACHE_DIR, OUTPUTS_DIR, PROJECT_ROOT,
    ZHIPU_API_KEY,
    ZHIPU_EMBEDDING_MODEL, ZHIPU_EMBEDDING_DIM,
    FAISS_INDEX_PATH, CHUNKS_DATA_PATH, EMBEDDING_CONFIG,
    SIMILARITY_THRESHOLD
)


def filter_relevant_results(results: List[Dict], threshold: float) -> Dict[str, Any]:
    """按阈值过滤检索结果，保留原始 top-k 语义，只做相关性筛选与重编号。

    纯函数，不依赖 FAISS / 网络 / API，便于单元测试。
    """
    relevant = [r for r in results if r["score"] >= threshold]
    max_score = max((r["score"] for r in results), default=0.0)
    for i, r in enumerate(relevant, 1):
        r["rank"] = i  # 保留结果重新编号
    return {
        "results": relevant,
        "total_results": len(relevant),
        "max_score": max_score,
        "relevant": max_score >= threshold,
        "threshold": threshold,
    }


class MedicalRAGEngine:
    """基于FAISS的医疗文档RAG引擎（智谱 embedding-2）"""

    # 分块配置（与原 EvoAgentX RAG_CONFIG 保持一致）
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    TOP_K = 5

    def __init__(self):
        """初始化医疗RAG引擎"""
        self.logger = logging.getLogger(__name__)

        # 智谱 Embedding 客户端
        self.embed_client = ZhipuAI(api_key=ZHIPU_API_KEY)
        self.embedding_model = ZHIPU_EMBEDDING_MODEL
        self.embedding_dim = ZHIPU_EMBEDDING_DIM

        # FAISS 索引（懒加载）
        self.faiss_index = None
        self.chunks_data: List[Dict[str, Any]] = []

        # 简单的 embedding 缓存，避免重复请求
        self._embed_cache: Dict[str, List[float]] = {}

        self.corpus_name = "medical_cases"
        self.is_indexed = False

        self.logger.info("医疗RAG引擎初始化完成（智谱 embedding-2）")

    # ============ 索引构建 ============

    def index_medical_documents(self, force_reindex: bool = False) -> bool:
        """索引医学PDF文档"""
        try:
            # 检查是否已经建立索引
            if not force_reindex and self._load_existing_index():
                self.logger.info("发现现有索引，跳过重建")
                self.is_indexed = True
                return True

            # 获取PDF文件列表
            pdf_files = sorted(list(DATA_DIR.glob("*.pdf")))
            if not pdf_files:
                self.logger.warning(f"在 {DATA_DIR} 中未找到PDF文件")
                return False

            self.logger.info(f"开始索引 {len(pdf_files)} 个医学PDF文档...")

            # 处理所有PDF，得到 chunks
            all_chunks: List[Dict[str, Any]] = []
            for pdf_file in pdf_files:
                self.logger.info(f"处理文档: {pdf_file.name}")
                file_chunks = self._process_pdf_file(pdf_file)
                all_chunks.extend(file_chunks)

            if not all_chunks:
                self.logger.error("未提取到任何文本块")
                return False

            self.logger.info(f"共 {len(all_chunks)} 个文本块，开始向量化...")

            # 批量生成 embedding
            texts = [c["text"] for c in all_chunks]
            embeddings = self._batch_embed(texts)

            if embeddings is None or len(embeddings) != len(all_chunks):
                self.logger.error("向量化失败")
                return False

            # 构建 FAISS 索引（Inner Product + 归一化 = 余弦相似度）
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(vectors)

            # 保存索引和 chunks 元数据
            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(CHUNKS_DATA_PATH, "wb") as f:
                pickle.dump(all_chunks, f)

            self.faiss_index = index
            self.chunks_data = all_chunks
            self.is_indexed = True

            self.logger.info(f"成功索引 {len(all_chunks)} 个文档块，维度 {self.embedding_dim}")
            return True

        except Exception as e:
            self.logger.error(f"索引文档时发生错误: {str(e)}")
            return False

    def _process_pdf_file(self, pdf_file: Path) -> List[Dict[str, Any]]:
        """处理单个PDF文件，返回 chunks 字典列表"""
        chunks = []
        try:
            with open(pdf_file, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

            if not text.strip():
                self.logger.warning(f"无法从 {pdf_file.name} 提取文本")
                return chunks

            text_chunks = self._chunk_text(text, self.CHUNK_SIZE, self.CHUNK_OVERLAP)

            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunks.append({
                        "chunk_id": f"{pdf_file.stem}_{i}",
                        "text": chunk_text,
                        "metadata": {
                            "doc_id": pdf_file.stem,
                            "corpus_id": self.corpus_name,
                            "chunk_index": i,
                            "file_name": pdf_file.name,
                            "file_path": str(pdf_file)
                        }
                    })
        except Exception as e:
            self.logger.error(f"处理PDF文件 {pdf_file} 时出错: {str(e)}")
        return chunks

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """文本分块"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            chunk = text[start:end]
            last_sentence_end = max(
                chunk.rfind('。'), chunk.rfind('!'), chunk.rfind('？'),
                chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?')
            )
            if last_sentence_end > chunk_size // 2:
                end = start + last_sentence_end + 1

            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _load_existing_index(self) -> bool:
        """加载现有索引"""
        try:
            if not (Path(FAISS_INDEX_PATH).exists() and Path(CHUNKS_DATA_PATH).exists()):
                return False
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_DATA_PATH, "rb") as f:
                self.chunks_data = pickle.load(f)
            self.logger.info(f"加载现有索引: {len(self.chunks_data)} 个文档块")
            return True
        except Exception as e:
            self.logger.warning(f"加载现有索引失败: {e}")
            return False

    # ============ Embedding 调用 ============

    def _batch_embed(self, texts: List[str], batch_size: int = 16) -> Any:
        """批量调用智谱 Embedding API"""
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 去掉两端空白，避免空字符串
            batch = [t.strip() or "空" for t in batch]
            try:
                resp = self.embed_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                for item in resp.data:
                    all_vectors.append(item.embedding)
                self.logger.info(f"已向量化 {min(i + batch_size, len(texts))}/{len(texts)}")
            except Exception as e:
                self.logger.error(f"Embedding 批次 {i} 失败: {e}")
                return None
            # 简单限速，避免触发 API 限流
            time.sleep(0.2)
        return all_vectors

    def _embed_query(self, text: str) -> Any:
        """单条查询向量化（带缓存）"""
        if text in self._embed_cache:
            return self._embed_cache[text]
        try:
            resp = self.embed_client.embeddings.create(
                model=self.embedding_model,
                input=text.strip() or "空"
            )
            vec = resp.data[0].embedding
            self._embed_cache[text] = vec
            return vec
        except Exception as e:
            self.logger.error(f"查询向量化失败: {e}")
            return None

    # ============ 检索 ============

    def search_similar_cases(self, query_text: str, top_k: int = None, threshold: float = None) -> Dict[str, Any]:
        """搜索相似医学病例"""
        try:
            if not self.is_indexed:
                # 自动加载索引
                if not self._load_existing_index():
                    return {
                        "status": "error",
                        "error": "RAG引擎尚未建立索引",
                        "results": []
                    }
                self.is_indexed = True

            top_k = top_k or self.TOP_K
            query_vec = self._embed_query(query_text)
            if query_vec is None:
                return {"status": "error", "error": "查询向量化失败", "results": []}

            query_arr = np.array([query_vec], dtype=np.float32)
            faiss.normalize_L2(query_arr)
            scores, indices = self.faiss_index.search(query_arr, top_k)

            formatted_results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx < 0 or idx >= len(self.chunks_data):
                    continue
                chunk = self.chunks_data[idx]
                formatted_results.append({
                    "rank": rank,
                    "content": chunk["text"],
                    "score": float(score),
                    "source": chunk["metadata"].get("file_path", ""),
                    "document_title": chunk["metadata"].get("file_name", ""),
                    "chunk_id": chunk["chunk_id"],
                    "metadata": {
                        "chunk_index": chunk["metadata"].get("chunk_index", 0),
                        "doc_id": chunk["metadata"].get("doc_id", "")
                    }
                })

            # 相关性过滤：剔除低于阈值的低分结果，避免无关内容喂给模型导致编造诊断
            threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
            filtered = filter_relevant_results(formatted_results, threshold)

            return {
                "status": "success",
                "query": query_text,
                "total_results": filtered["total_results"],
                "results": filtered["results"],
                "max_score": filtered["max_score"],
                "relevant": filtered["relevant"],
                "threshold": filtered["threshold"],
                "search_metadata": {
                    "corpus_name": self.corpus_name,
                    "top_k": top_k,
                    "embedding_model": self.embedding_model,
                    "embedding_dim": self.embedding_dim,
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            self.logger.error(f"检索过程中发生错误: {str(e)}")
            return {"status": "error", "error": str(e), "results": []}

    # ============ 信息查询 ============

    def get_corpus_info(self) -> Dict[str, Any]:
        """获取语料库信息"""
        try:
            return {
                "corpus_name": self.corpus_name,
                "is_indexed": self.is_indexed,
                "storage_path": str(CACHE_DIR),
                "total_documents": len(list(DATA_DIR.glob("*.pdf"))),
                "total_chunks": len(self.chunks_data),
                "rag_config": {
                    "chunk_size": self.CHUNK_SIZE,
                    "chunk_overlap": self.CHUNK_OVERLAP,
                    "embedding_model": self.embedding_model,
                    "embedding_dim": self.embedding_dim,
                    "top_k": self.TOP_K
                }
            }
        except Exception as e:
            self.logger.error(f"获取语料库信息时出错: {str(e)}")
            return {}

    def save_search_results(self, results: Dict[str, Any], query: str) -> str:
        """保存搜索结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_search_{timestamp}.json"
            filepath = OUTPUTS_DIR / "results" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            save_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "engine_info": self.get_corpus_info(),
                "search_results": results
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"搜索结果已保存到: {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"保存搜索结果时出错: {str(e)}")
            return ""


def main():
    """测试医疗RAG引擎"""
    print("🏥 初始化医疗RAG引擎...")
    medical_rag = MedicalRAGEngine()

    print("📚 正在索引医学文档...")
    success = medical_rag.index_medical_documents()
    if not success:
        print("❌ 文档索引失败")
        return
    print("✅ 文档索引完成")

    test_queries = [
        "患者出现头痛症状",
        "血压升高伴视物模糊",
        "胸痛呼吸困难心电图异常"
    ]
    for query in test_queries:
        print(f"\n🔍 搜索: {query}")
        results = medical_rag.search_similar_cases(query, top_k=3)
        if results["status"] == "success":
            print(f"找到 {results['total_results']} 个相似病例:")
            for result in results["results"]:
                print(f"  - 排名 {result['rank']}: {result['document_title']}")
                print(f"    相似度: {result['score']:.4f}")
                print(f"    内容: {result['content'][:100]}...")
        else:
            print(f"❌ 搜索失败: {results.get('error', '未知错误')}")

    print("\n📊 语料库信息:")
    print(json.dumps(medical_rag.get_corpus_info(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
