"""medical_conversation 会话状态机的纯本地单元测试。

不联网、不调模型，只验证问诊顺序、跳过值、年龄校验、危险信号、确认与重分析等逻辑。
运行：python -m unittest test_medical_conversation
"""

import unittest

import medical_conversation as mc


def _drive(state, answers):
    """依次把一串用户回答喂给 handle_user_message，返回最终状态。"""
    for a in answers:
        state = mc.handle_user_message(state, a)
    return state


# 覆盖全部 9 个问诊步骤的合法回答，用于走完整个 intake 流程。
INTAKE_ANSWERS = [
    "发热、咳嗽 3 天",   # 主诉
    "58 岁，男性",       # 年龄和性别
    "3 天",              # 起病/病程
    "无",                # 伴随症状
    "高血压 10 年",      # 既往史
    "无",                # 正在使用的药物
    "无",                # 过敏史
    "无",                # 生命体征/检查结果
    "无",                # 其他补充
]


class TestIntakeFlow(unittest.TestCase):
    def test_create_session_starts_at_intake(self):
        state = mc.create_session()
        self.assertEqual(state["stage"], mc.STAGE_INTAKE)
        self.assertEqual(state["step_index"], 0)
        self.assertEqual(state["patient"], {})
        # 欢迎语 + 第一个问题都已写入聊天记录
        self.assertGreaterEqual(len(state["history"]), 2)

    def test_full_intake_reaches_confirm(self):
        state = _drive(mc.create_session(), INTAKE_ANSWERS)
        self.assertEqual(state["stage"], mc.STAGE_CONFIRM)

        patient = state["patient"]
        self.assertEqual(patient["chief_complaint"], "发热、咳嗽 3 天")
        self.assertEqual(patient["age"], "58")
        self.assertEqual(patient["gender"], "男性")
        self.assertEqual(patient["duration"], "3 天")
        self.assertEqual(patient["history"], "高血压 10 年")

    def test_step_index_advances_one_per_answer(self):
        state = mc.create_session()
        state = mc.handle_user_message(state, "发热咳嗽")
        self.assertEqual(state["step_index"], 1)
        state = mc.handle_user_message(state, "58 岁，男性")
        self.assertEqual(state["step_index"], 2)


class TestSkipAndUnknown(unittest.TestCase):
    def test_skip_values_normalise_to_unknown(self):
        state = _drive(mc.create_session(), ["发热", "58 岁，男性", "不清楚"])
        self.assertEqual(state["patient"]["duration"], mc.UNKNOWN_VALUE)

    def test_optional_field_accepts_none(self):
        state = _drive(mc.create_session(), ["发热", "58 岁，男性", "无"])
        self.assertEqual(state["patient"]["duration"], mc.UNKNOWN_VALUE)


class TestAgeValidation(unittest.TestCase):
    def _to_demographics_step(self):
        state = mc.create_session()
        return mc.handle_user_message(state, "发热")

    def test_natural_answer_parsed(self):
        state = mc.handle_user_message(self._to_demographics_step(), "58 岁，男性")
        self.assertEqual(state["patient"]["age"], "58")
        self.assertEqual(state["patient"]["gender"], "男性")

    def test_age_out_of_range_rejected(self):
        state = self._to_demographics_step()
        state = mc.handle_user_message(state, "300 岁，男性")
        # 未通过校验：step_index 不前进，patient 里没有 age
        self.assertNotIn("age", state["patient"])
        self.assertEqual(state["step_index"], 1)
        self.assertIn("0 到 150", state["history"][-1]["content"])

    def test_missing_age_or_gender_asks_again(self):
        state = self._to_demographics_step()
        state = mc.handle_user_message(state, "男性")
        self.assertNotIn("age", state["patient"])
        self.assertEqual(state["step_index"], 1)


class TestConfirmAndAnalyze(unittest.TestCase):
    def test_begin_analysis_only_from_confirm(self):
        intake_state = mc.create_session()
        still_intake = mc.begin_analysis(intake_state)
        self.assertEqual(still_intake["stage"], mc.STAGE_INTAKE)

        confirm_state = _drive(mc.create_session(), INTAKE_ANSWERS)
        analyzing = mc.begin_analysis(confirm_state)
        self.assertEqual(analyzing["stage"], mc.STAGE_ANALYZING)

    def test_confirm_extra_appends_user_input(self):
        confirm_state = _drive(mc.create_session(), INTAKE_ANSWERS)
        updated = mc.handle_user_message(confirm_state, "家族史：父亲高血压")
        self.assertEqual(updated["stage"], mc.STAGE_CONFIRM)
        self.assertEqual(updated["patient"]["extra"], "家族史：父亲高血压")


class TestEmergency(unittest.TestCase):
    def test_detect_emergency_keywords(self):
        self.assertEqual(mc.detect_emergency("我突然胸痛"), ["胸痛"])
        self.assertEqual(mc.detect_emergency("喘不过气"), ["呼吸困难"])
        self.assertEqual(mc.detect_emergency("我最近睡眠不好"), [])

    def test_emergency_stops_intake_and_no_analysis(self):
        state = mc.handle_user_message(mc.create_session(), "我现在胸痛得厉害")
        self.assertEqual(state["stage"], mc.STAGE_EMERGENCY)
        self.assertTrue(state["emergency_reasons"])
        self.assertEqual(state.get("analysis"), {})  # 未触发任何分析

    def test_emergency_detected_in_later_stage(self):
        state = _drive(mc.create_session(), ["发热", "58 岁，男性"])
        state = mc.handle_user_message(state, "呼吸困难")
        self.assertEqual(state["stage"], mc.STAGE_EMERGENCY)


class TestReset(unittest.TestCase):
    def test_create_session_is_fresh(self):
        s1 = mc.handle_user_message(mc.create_session(), "发热")
        s2 = mc.create_session()
        self.assertEqual(s2["step_index"], 0)
        self.assertEqual(s2["patient"], {})
        self.assertEqual(s2["stage"], mc.STAGE_INTAKE)


class TestReanalysis(unittest.TestCase):
    def _followup_state(self):
        state = mc.create_session()
        state["stage"] = mc.STAGE_FOLLOWUP
        state["last_followup"] = "复查发现血压升高"
        state["pending_followup"] = "复查发现血压升高"
        state["analysis"] = {"final_report": "示例报告"}
        return state

    def test_include_followup_in_reanalysis(self):
        state = mc.include_last_followup_in_reanalysis(self._followup_state())
        self.assertEqual(state["stage"], mc.STAGE_CONFIRM)
        self.assertIn("复查发现血压升高", state["followup_updates"])
        self.assertEqual(state["last_followup"], "")
        self.assertEqual(state["analysis"], {})

    def test_include_ignored_when_not_followup(self):
        state = mc.include_last_followup_in_reanalysis(mc.create_session())
        self.assertEqual(state["stage"], mc.STAGE_INTAKE)


class TestHelpers(unittest.TestCase):
    def test_history_to_chatbot_messages(self):
        state = mc.create_session()
        msgs = mc.history_to_chatbot(state)
        self.assertTrue(msgs)
        self.assertEqual(msgs[0]["role"], "assistant")  # 首条为助手欢迎语
        self.assertIn("content", msgs[0])

    def test_build_retrieval_query_excludes_labels(self):
        state = _drive(mc.create_session(), INTAKE_ANSWERS)
        query = mc.build_retrieval_query(state)
        # 检索查询只含真实症状内容，不含“患者/主诉”等标签词
        self.assertNotIn("患者", query)
        self.assertNotIn("主诉", query)
        self.assertIn("发热、咳嗽 3 天", query)


class TestGoBack(unittest.TestCase):
    def test_go_back_after_first_answer(self):
        state = mc.handle_user_message(mc.create_session(), "发热")
        self.assertEqual(state["step_index"], 1)
        out = mc.go_back(state)
        self.assertEqual(out["step_index"], 0)
        self.assertNotIn("chief_complaint", out["patient"])
        self.assertIn("请重新回答上一项", out["history"][-1]["content"])

    def test_go_back_at_first_question_is_noop(self):
        state = mc.create_session()
        out = mc.go_back(state)
        self.assertEqual(out["step_index"], 0)
        self.assertIn("已经是第一项提问", out["history"][-1]["content"])

    def test_go_back_clears_demographics_both_keys(self):
        state = _drive(mc.create_session(), ["发热", "58 岁，男性"])
        self.assertEqual(state["step_index"], 2)
        out = mc.go_back(state)
        self.assertEqual(out["step_index"], 1)
        self.assertNotIn("age", out["patient"])
        self.assertNotIn("gender", out["patient"])
        self.assertIn("chief_complaint", out["patient"])

    def test_go_back_from_confirm_returns_to_last_step(self):
        state = _drive(mc.create_session(), INTAKE_ANSWERS)
        self.assertEqual(state["stage"], mc.STAGE_CONFIRM)
        out = mc.go_back(state)
        self.assertEqual(out["stage"], mc.STAGE_INTAKE)
        self.assertEqual(out["step_index"], len(mc.INTAKE_STEPS) - 1)
        self.assertNotIn("extra", out["patient"])

    def test_text_keyword_triggers_go_back(self):
        state = mc.handle_user_message(mc.create_session(), "发热")
        state = mc.handle_user_message(state, "返回上一题")
        self.assertEqual(state["step_index"], 0)

    def test_go_back_ignored_outside_intake(self):
        state = mc.create_session()
        state["stage"] = mc.STAGE_FOLLOWUP
        state["step_index"] = 3
        out = mc.go_back(state)
        self.assertEqual(out["stage"], mc.STAGE_FOLLOWUP)
        self.assertEqual(out["step_index"], 3)


if __name__ == "__main__":
    unittest.main()
