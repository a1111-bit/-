"""工作流文本解析辅助方法的单元测试（只测纯字符串处理，不联网、不调模型、不建索引）。

关键点：MedicalWorkflowExecutor.__init__ 会初始化 RAG 引擎 / 工具 / 智谱客户端（重、需 API Key、需联网），
而这些待测方法都不使用 self 的任何成员，因此用 object.__new__(MedicalWorkflowExecutor) 构造"空壳"实例，
绕过 __init__，只测那 6 个纯文本解析方法。

运行方式（在 app 目录下）：
    python -m unittest test_workflow_helpers -v
"""

import sys
import unittest
from pathlib import Path

# 确保能 import 同目录下的模块（从任意目录运行都能找到 app 下的文件）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evoagentx_medical_workflow import MedicalWorkflowExecutor


def _bare_executor():
    """绕过 __init__ 构造一个空壳执行器，避免初始化 RAG/工具/LLM。"""
    return object.__new__(MedicalWorkflowExecutor)


class TestExtractDiagnoses(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_basic_extraction_stops_at_tests(self):
        content = "可能病因\n- 急性支气管炎\n- 肺炎\n检查建议\n- 血常规"
        self.assertEqual(self.ex._extract_diagnoses(content), "- 急性支气管炎\n- 肺炎")

    def test_no_trigger_falls_back(self):
        content = "这是一段只描述症状的普通文本"
        self.assertEqual(self.ex._extract_diagnoses(content), "需要进一步分析确定诊断方向")


class TestExtractTests(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_basic_extraction_stops_at_treatment(self):
        content = "检查建议\n- 血常规\n- 胸部X线\n治疗建议\n- 抗生素"
        self.assertEqual(self.ex._extract_tests(content), "- 血常规\n- 胸部X线")

    def test_no_trigger_falls_back(self):
        content = "普通文本内容"
        self.assertEqual(self.ex._extract_tests(content), "建议常规实验室检查和影像学检查")


class TestExtractSummary(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_basic_extraction_stops_at_next_heading(self):
        content = "## 执行摘要\n- 病例概述\n- 主要发现\n## 1. 病例信息\n- 详情"
        self.assertEqual(self.ex._extract_summary(content), "- 病例概述\n- 主要发现")

    def test_no_trigger_falls_back(self):
        content = "这是一段普通文本"
        self.assertEqual(self.ex._extract_summary(content), "请查看完整报告获取详细信息")


class TestExtractGuidance(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_short_content_returned_as_is(self):
        self.assertEqual(self.ex._extract_guidance("简短指导"), "简短指导")

    def test_long_content_truncated_with_ellipsis(self):
        out = self.ex._extract_guidance("A" * 600)
        self.assertEqual(len(out), 503)  # 500 + "..."
        self.assertTrue(out.endswith("..."))
        self.assertTrue(out.startswith("A" * 500))


class TestParseDiagnosesList(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_strips_numbering_and_caps_at_five(self):
        text = (
            "1. 急性支气管炎\n"
            "2-社区获得性肺炎\n"
            "3、慢性阻塞性肺疾病\n"
            "4 支气管哮喘\n"
            "5 过敏性鼻炎\n"
            "6 上呼吸道感染"
        )
        self.assertEqual(
            self.ex._parse_diagnoses_list(text),
            ["急性支气管炎", "社区获得性肺炎", "慢性阻塞性肺疾病", "支气管哮喘", "过敏性鼻炎"],
        )

    def test_filters_short_entries(self):
        text = "1. 肺炎\n2. 感冒\n3. 支气管炎"
        self.assertEqual(self.ex._parse_diagnoses_list(text), ["支气管炎"])

    def test_skips_hash_heading_lines(self):
        text = "# 可能病因\n1. 支气管炎"
        self.assertEqual(self.ex._parse_diagnoses_list(text), ["支气管炎"])


class TestFormatRagResults(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_empty_results_returns_placeholder(self):
        self.assertEqual(self.ex._format_rag_results({"results": []}), "未找到相关医学案例。")
        self.assertEqual(self.ex._format_rag_results({}), "未找到相关医学案例。")

    def test_normal_result_formatted(self):
        results = {
            "results": [{"rank": 1, "score": 0.85, "content": "短内容", "document_title": "docA"}]
        }
        out = self.ex._format_rag_results(results)
        self.assertIn("检索到的相关医学案例:", out)
        self.assertIn("案例 1 (相似度: 0.8500)", out)
        self.assertIn("来源: docA", out)
        self.assertIn("内容: 短内容", out)

    def test_long_content_truncated(self):
        long_content = "长" * 500
        results = {
            "results": [{"rank": 1, "score": 0.9, "content": long_content, "document_title": "docB"}]
        }
        out = self.ex._format_rag_results(results)
        self.assertIn("长" * 400 + "...", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
