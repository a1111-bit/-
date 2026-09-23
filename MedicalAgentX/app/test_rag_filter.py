"""
RAG 检索相关性过滤逻辑的单元测试（仅测纯函数 filter_relevant_results，无需网络/API 调用）。

运行方式（在 app 目录下）：
    python test_rag_filter.py
或：
    python -m unittest test_rag_filter -v
"""

import sys
import unittest
from pathlib import Path

# 确保能 import 同目录下的模块（从任意目录运行都能找到 app 下的文件）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evoagentx_medical_engine import filter_relevant_results


def _make_result(rank, score):
    """构造一条检索结果，结构与 search_similar_cases 中的 formatted_results 一致。"""
    return {"rank": rank, "score": score, "content": f"chunk{rank}", "document_title": "doc"}


class TestFilterRelevantResults(unittest.TestCase):
    def test_all_below_threshold(self):
        """全部低于阈值 → 全过滤，relevant=False，total_results=0"""
        results = [_make_result(1, 0.10), _make_result(2, 0.20), _make_result(3, 0.05)]
        out = filter_relevant_results(results, 0.35)
        self.assertEqual(out["total_results"], 0)
        self.assertEqual(out["results"], [])
        self.assertFalse(out["relevant"])
        self.assertAlmostEqual(out["max_score"], 0.20)
        self.assertEqual(out["threshold"], 0.35)

    def test_partial_kept_renumbered(self):
        """部分保留 → 结果按原顺序重编号为 1..N"""
        results = [
            _make_result(1, 0.10),
            _make_result(2, 0.60),
            _make_result(3, 0.40),
            _make_result(4, 0.05),
        ]
        out = filter_relevant_results(results, 0.35)
        self.assertEqual(out["total_results"], 2)
        self.assertTrue(out["relevant"])
        self.assertAlmostEqual(out["max_score"], 0.60)
        self.assertEqual([r["rank"] for r in out["results"]], [1, 2])
        self.assertEqual([r["score"] for r in out["results"]], [0.60, 0.40])

    def test_boundary_equal_threshold_kept(self):
        """边界值 score == 阈值 → 保留（>=）"""
        out = filter_relevant_results([_make_result(1, 0.35)], 0.35)
        self.assertEqual(out["total_results"], 1)
        self.assertTrue(out["relevant"])
        self.assertEqual(out["results"][0]["score"], 0.35)

    def test_empty_list(self):
        """空列表 → max_score=0，relevant=False，total_results=0"""
        out = filter_relevant_results([], 0.35)
        self.assertEqual(out["total_results"], 0)
        self.assertEqual(out["results"], [])
        self.assertFalse(out["relevant"])
        self.assertEqual(out["max_score"], 0.0)
        self.assertEqual(out["threshold"], 0.35)


if __name__ == "__main__":
    unittest.main(verbosity=2)
