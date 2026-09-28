"""工具结果序列化 / prompt 填充的单元测试（纯函数，不联网、不调模型、不建索引）。

与 test_workflow_helpers.py 相同，用 object.__new__(MedicalWorkflowExecutor) 构造空壳实例，
绕过 __init__，只测不依赖 self 成员的方法。
"""

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evoagentx_medical_workflow import MedicalWorkflowExecutor


def _bare_executor():
    ex = object.__new__(MedicalWorkflowExecutor)
    ex.logger = logging.getLogger("test_tool_format")  # _fill_prompt 在字段缺失分支会用到 logger
    ex.logger.setLevel(logging.ERROR)  # 避免把中文 warning 打到控制台（GBK 会乱码）
    return ex


class TestFormatToolResults(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_none_returns_placeholder(self):
        self.assertEqual(
            self.ex._format_tool_results(None),
            "（工具未执行或不可用，请基于通用医学知识补充。）",
        )

    def test_normal_result_serialized_as_json(self):
        result = {
            "comprehensive_results": {
                "disease_information": {"高血压": "降压治疗"},
                "search_summary": "命中 1 个疾病",
            }
        }
        out = self.ex._format_tool_results(result)
        self.assertIn("高血压", out)
        self.assertIn("search_summary", out)

    def test_long_result_truncated(self):
        result = {"comprehensive_results": {"search_summary": "x" * 2000}}
        out = self.ex._format_tool_results(result)
        self.assertTrue(out.endswith("\n…（结果过长已截断）"))
        self.assertEqual(len(out), 1500 + len("\n…（结果过长已截断）"))


class TestFillPrompt(unittest.TestCase):
    def setUp(self):
        self.ex = _bare_executor()

    def test_all_fields_present(self):
        out = self.ex._fill_prompt("症状：{symptom_text}", {"symptom_text": "头痛"})
        self.assertEqual(out, "症状：头痛")

    def test_missing_field_filled_with_empty(self):
        out = self.ex._fill_prompt("症状：{symptom_text}｜工具：{tool_results}", {"symptom_text": "头痛"})
        self.assertEqual(out, "症状：头痛｜工具：")


if __name__ == "__main__":
    unittest.main(verbosity=2)
