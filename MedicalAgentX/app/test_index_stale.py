"""索引过期检测 index_is_stale 的单元测试（只碰临时目录，不联网、不初始化 ZhipuAI）。

用 object.__new__(MedicalRAGEngine) 构造空壳，绕过 __init__；并把模块级 DATA_DIR /
FAISS_INDEX_PATH 临时替换为测试目录，避免读写真实数据。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evoagentx_medical_engine as engine


class TestIndexIsStale(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "data"
        self.data_dir.mkdir()
        self.index_path = Path(self.tmp.name) / "cache" / "faiss_index.index"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._orig_data = engine.DATA_DIR
        self._orig_index = engine.FAISS_INDEX_PATH
        engine.DATA_DIR = self.data_dir
        engine.FAISS_INDEX_PATH = str(self.index_path)

        self.rag = object.__new__(engine.MedicalRAGEngine)

    def tearDown(self):
        engine.DATA_DIR = self._orig_data
        engine.FAISS_INDEX_PATH = self._orig_index
        self.tmp.cleanup()

    def _touch_pdf(self, name, mtime):
        p = self.data_dir / name
        p.write_text("x", encoding="utf-8")
        os.utime(p, (mtime, mtime))
        return p

    def test_no_pdfs_not_stale(self):
        self.assertFalse(self.rag.index_is_stale())

    def test_no_index_yet_is_stale(self):
        self._touch_pdf("a.pdf", 100)
        self.assertTrue(self.rag.index_is_stale())

    def test_pdf_newer_than_index_is_stale(self):
        self._touch_pdf("a.pdf", 200)
        self.index_path.write_text("dummy", encoding="utf-8")
        os.utime(self.index_path, (100, 100))
        self.assertTrue(self.rag.index_is_stale())

    def test_pdf_older_than_index_not_stale(self):
        self._touch_pdf("a.pdf", 100)
        self.index_path.write_text("dummy", encoding="utf-8")
        os.utime(self.index_path, (200, 200))
        self.assertFalse(self.rag.index_is_stale())


if __name__ == "__main__":
    unittest.main(verbosity=2)
