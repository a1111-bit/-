"""纯本地的多轮医疗问诊会话状态机。

本模块不调用模型、检索或网络服务，因此可以独立测试。网页层负责把状态保存
到 ``gr.State``，并在 ``confirm`` / ``followup`` 阶段按需调用工作流。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional


STAGE_INTAKE = "intake"
STAGE_CONFIRM = "confirm"
STAGE_ANALYZING = "analyzing"
STAGE_FOLLOWUP = "followup"
STAGE_EMERGENCY = "emergency"

UNKNOWN_VALUE = "未提供"


INTAKE_STEPS = [
    {
        "key": "chief_complaint",
        "label": "主诉",
        "prompt": "请先描述最让患者困扰的不适或就诊原因，例如“发热、咳嗽 3 天”。",
        "required": True,
    },
    {
        "key": "demographics",
        "label": "年龄和性别",
        "prompt": "患者的年龄和性别是什么？例如“58 岁，男性”。如确实不清楚，可回答“不清楚”。",
        "required": False,
    },
    {
        "key": "duration",
        "label": "起病/病程",
        "prompt": "这些症状从什么时候开始？是突然出现还是逐渐加重？",
        "required": False,
    },
    {
        "key": "accompanying",
        "label": "伴随症状",
        "prompt": "还伴随哪些症状？可补充严重程度、发热、疼痛部位、呕吐等；没有可回答“无”。",
        "required": False,
    },
    {
        "key": "history",
        "label": "既往史",
        "prompt": "患者有高血压、糖尿病、心肺疾病、手术史等既往病史吗？",
        "required": False,
    },
    {
        "key": "medications",
        "label": "正在使用的药物",
        "prompt": "目前正在使用哪些药物或保健品？没有可回答“无”。",
        "required": False,
    },
    {
        "key": "allergy",
        "label": "过敏史",
        "prompt": "是否有药物、食物或其他过敏史？没有可回答“无”。",
        "required": False,
    },
    {
        "key": "vitals_tests",
        "label": "生命体征/检查结果",
        "prompt": "是否有体温、血压、血糖、化验单、心电图或影像检查结果？没有可回答“无”。",
        "required": False,
    },
    {
        "key": "extra",
        "label": "其他补充",
        "prompt": "还有其他需要补充的信息吗，例如家族史、妊娠情况或近期特殊经历？没有可回答“无”。",
        "required": False,
    },
]


EMERGENCY_PATTERNS = {
    "胸痛": "胸痛",
    "呼吸困难": "呼吸困难",
    "喘不过气": "呼吸困难",
    "无法呼吸": "呼吸困难",
    "意识不清": "意识异常",
    "昏迷": "意识异常",
    "抽搐": "抽搐",
    "单侧无力": "疑似卒中表现",
    "一侧无力": "疑似卒中表现",
    "口角歪": "疑似卒中表现",
    "言语不清": "疑似卒中表现",
    "说话不清": "疑似卒中表现",
    "呕血": "消化道出血",
    "便血": "消化道出血",
    "黑便": "消化道出血",
    "大出血": "大出血",
    "大量出血": "大出血",
    "自杀": "自伤风险",
    "自伤": "自伤风险",
    "想死": "自伤风险",
}

SKIP_RESPONSES = {"无", "没有", "不清楚", "不知道", "跳过", "未提供", "不详"}

# 在输入框里输入这些词会触发「返回上一题」（与按钮等价）
GO_BACK_KEYWORDS = {"返回上一题", "上一题", "重新回答上一题", "返回上一步"}


@dataclass
class ConsultationSession:
    """当前浏览器标签页的一次问诊；转换为 dict 后可安全放入 gr.State。"""

    stage: str = STAGE_INTAKE
    step_index: int = 0
    patient: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    followup_updates: List[str] = field(default_factory=list)
    last_followup: str = ""
    pending_followup: str = ""
    emergency_reasons: List[str] = field(default_factory=list)
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "ConsultationSession":
        if not value:
            return cls()
        allowed = {name: value.get(name) for name in cls.__dataclass_fields__ if name in value}
        return cls(**allowed)


def _assistant(session: ConsultationSession, content: str) -> None:
    session.history.append({"role": "assistant", "content": content})


def _user(session: ConsultationSession, content: str) -> None:
    session.history.append({"role": "user", "content": content})


def _normalise_optional_value(text: str) -> str:
    return UNKNOWN_VALUE if text.strip().lower() in SKIP_RESPONSES else text.strip()


def _current_step(session: ConsultationSession) -> Dict[str, Any]:
    return INTAKE_STEPS[min(session.step_index, len(INTAKE_STEPS) - 1)]


def _append_next_question_or_confirm(session: ConsultationSession) -> None:
    if session.step_index < len(INTAKE_STEPS):
        _assistant(session, _current_step(session)["prompt"])
        return

    session.stage = STAGE_CONFIRM
    _assistant(
        session,
        "信息已收集完成，请核对下面的病例摘要。确认无误后点击“开始完整分析”；"
        "也可以继续在聊天框补充信息。\n\n" + build_case_summary(session.to_dict()),
    )


def create_session() -> Dict[str, Any]:
    """创建一段新会话并写入首条欢迎语。"""
    session = ConsultationSession()
    _assistant(
        session,
        "你好，我会通过几轮简短提问收集病例资料，再生成医疗辅助分析报告。"
        "如出现紧急症状，请立即线下就医。",
    )
    _append_next_question_or_confirm(session)
    return session.to_dict()


def detect_emergency(text: str) -> List[str]:
    """返回输入中命中的危险信号名称；规则保守，宁可优先提示急诊。"""
    return list(dict.fromkeys(label for pattern, label in EMERGENCY_PATTERNS.items() if pattern in text))


def _enter_emergency(session: ConsultationSession, reasons: List[str]) -> None:
    session.stage = STAGE_EMERGENCY
    session.emergency_reasons = reasons
    session.pending_followup = ""
    reason_text = "、".join(reasons)
    _assistant(
        session,
        "## ⚠️ 请立即寻求线下紧急医疗帮助\n\n"
        f"系统识别到可能的危险信号：{reason_text}。请尽快前往急诊或拨打当地急救电话，"
        "不要等待在线问诊或 AI 报告。",
    )


def _extract_demographics(text: str) -> Dict[str, str]:
    cleaned = text.strip()
    if cleaned.lower() in SKIP_RESPONSES:
        return {"age": UNKNOWN_VALUE, "gender": UNKNOWN_VALUE}

    age_match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", cleaned)
    age = age_match.group(1) if age_match else ""
    if age and not 0 <= int(age) <= 150:
        return {"error": "年龄需在 0 到 150 之间，请重新输入，例如“58 岁，男性”。"}

    gender = ""
    if "女性" in cleaned or "女" in cleaned:
        gender = "女性"
    elif "男性" in cleaned or "男" in cleaned:
        gender = "男性"

    missing = []
    if not age:
        missing.append("年龄")
    if not gender:
        missing.append("性别")
    if missing:
        return {"error": "还缺少" + "和".join(missing) + "，请按“58 岁，男性”的格式补充；不清楚可回答“不清楚”。"}
    return {"age": age, "gender": gender}


def _record_intake_answer(session: ConsultationSession, answer: str) -> Optional[str]:
    step = _current_step(session)
    key = step["key"]

    if key == "chief_complaint":
        if answer.strip().lower() in SKIP_RESPONSES:
            return "主诉是生成分析所必需的信息，请描述患者当前最主要的不适。"
        session.patient[key] = answer.strip()
    elif key == "demographics":
        demographics = _extract_demographics(answer)
        if "error" in demographics:
            return demographics["error"]
        session.patient.update(demographics)
    else:
        session.patient[key] = _normalise_optional_value(answer)

    session.step_index += 1
    return None


def _clear_step(session: ConsultationSession, index: int) -> None:
    """清掉第 index 项提问已记录的答案（demographics 要同时清 age 与 gender）。"""
    key = INTAKE_STEPS[index]["key"]
    if key == "demographics":
        session.patient.pop("age", None)
        session.patient.pop("gender", None)
    else:
        session.patient.pop(key, None)


def _go_back(session: ConsultationSession) -> bool:
    """把问诊退回到上一项提问并重新提问；返回是否成功退回。"""
    if session.stage == STAGE_CONFIRM:
        session.stage = STAGE_INTAKE  # 从确认页退回最后一项提问
    elif session.stage != STAGE_INTAKE:
        return False  # 分析中 / 追问 / 急诊等阶段不可返回

    if session.step_index <= 0:
        _assistant(session, "这已经是第一项提问了，无法再返回上一题。")
        return False

    target = session.step_index - 1
    _clear_step(session, target)
    session.step_index = target
    _assistant(session, "好的，请重新回答上一项：\n\n" + INTAKE_STEPS[target]["prompt"])
    return True


def go_back(state: Dict[str, Any]) -> Dict[str, Any]:
    """「返回上一题」按钮入口：退回一步并重新提问。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    _go_back(session)
    return session.to_dict()


def handle_user_message(state: Optional[Dict[str, Any]], text: str) -> Dict[str, Any]:
    """处理一次用户输入，不包含模型调用。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    message = (text or "").strip()

    if not message:
        if session.stage == STAGE_INTAKE:
            _assistant(session, "请输入当前问题的回答；如确实不清楚，可以回答“不清楚”或“跳过”。")
        return session.to_dict()

    if session.stage == STAGE_EMERGENCY:
        _assistant(session, "当前会话已停止常规问诊。请优先获得线下紧急医疗帮助，或点击“新建病例”开始新的会话。")
        return session.to_dict()

    # 「返回上一题」文字指令：在当作普通回答之前拦截
    if session.stage in (STAGE_INTAKE, STAGE_CONFIRM) and message in GO_BACK_KEYWORDS:
        _go_back(session)
        return session.to_dict()

    _user(session, message)
    emergency_reasons = detect_emergency(message)
    if emergency_reasons:
        _enter_emergency(session, emergency_reasons)
        return session.to_dict()

    if session.stage == STAGE_INTAKE:
        validation_error = _record_intake_answer(session, message)
        if validation_error:
            _assistant(session, validation_error)
        else:
            _append_next_question_or_confirm(session)
        return session.to_dict()

    if session.stage == STAGE_CONFIRM:
        existing = session.patient.get("extra", "")
        if existing in ("", UNKNOWN_VALUE):
            session.patient["extra"] = message
        else:
            session.patient["extra"] = f"{existing}\n补充：{message}"
        _assistant(
            session,
            "已将这条内容补充到病例资料中。请确认病例摘要后点击“开始完整分析”。\n\n"
            + build_case_summary(session.to_dict()),
        )
        return session.to_dict()

    if session.stage == STAGE_FOLLOWUP:
        session.pending_followup = message
        session.last_followup = message
        return session.to_dict()

    _assistant(session, "系统正在处理当前分析，请稍候。")
    return session.to_dict()


def begin_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """仅在用户确认后进入分析阶段；实际工作流由网页层异步执行。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    if session.stage != STAGE_CONFIRM:
        return session.to_dict()
    session.stage = STAGE_ANALYZING
    session.last_error = ""
    _assistant(session, "已确认病例资料，正在执行医学资料检索和多智能体分析，请稍候。")
    return session.to_dict()


def record_analysis_result(state: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """记录一次成功（含无相关资料）的完整分析结果。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    session.analysis = deepcopy(result or {})
    session.stage = STAGE_FOLLOWUP
    session.pending_followup = ""
    session.last_followup = ""
    if result.get("low_relevance"):
        content = result.get("final_report") or "未检索到足够的相关医学资料。"
    else:
        summary = result.get("executive_summary") or result.get("final_report") or "分析已完成。"
        content = ("## 本次分析摘要\n\n" + summary +
                   "\n\n📄 完整报告见下方「完整报告」标签页。"
                   "你可以继续提问；如补充了新的症状或检查结果，可点击“将此补充纳入重新分析”。")
    _assistant(session, content)
    return session.to_dict()


def record_analysis_failure(state: Dict[str, Any], error: str = "") -> Dict[str, Any]:
    """失败时返回确认阶段，以便用户修正资料或重试。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    session.stage = STAGE_CONFIRM
    session.last_error = error
    _assistant(session, "完整分析暂时未能完成。病例资料仍已保留，请稍后点击“开始完整分析”重试。")
    return session.to_dict()


def record_followup_answer(state: Dict[str, Any], answer: str) -> Dict[str, Any]:
    session = ConsultationSession.from_dict(deepcopy(state))
    if session.stage == STAGE_FOLLOWUP and session.pending_followup:
        _assistant(session, answer or "暂时无法生成回答，请稍后重试或咨询线下医生。")
        session.pending_followup = ""
    return session.to_dict()


def include_last_followup_in_reanalysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """把用户明确选择的最新追问并入病例，并要求再次确认后再跑完整工作流。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    if session.stage != STAGE_FOLLOWUP or not session.last_followup:
        return session.to_dict()

    session.followup_updates.append(session.last_followup)
    session.analysis = {}
    session.pending_followup = ""
    session.last_followup = ""
    session.stage = STAGE_CONFIRM
    _assistant(
        session,
        "已将最新补充纳入病例。请核对更新后的信息，再点击“开始完整分析”生成新版报告。\n\n"
        + build_case_summary(session.to_dict()),
    )
    return session.to_dict()


def record_upload(state: Dict[str, Any], text: str) -> Dict[str, Any]:
    """把上传文件抽取的文本并入「生命体征/检查结果」栏，供用户核对。"""
    session = ConsultationSession.from_dict(deepcopy(state))
    text = (text or "").strip()
    if not text:
        _assistant(session, "未能从上传文件中提取到文字，请换用 PDF 或 TXT 文件。")
        return session.to_dict()
    existing = session.patient.get("vitals_tests", "")
    if existing in ("", UNKNOWN_VALUE):
        session.patient["vitals_tests"] = text
    else:
        session.patient["vitals_tests"] = f"{existing}\n上传资料：{text}"
    session.stage = STAGE_CONFIRM
    _assistant(
        session,
        "已将上传文件的内容并入「生命体征/检查结果」。请核对病例摘要后点击“开始完整分析”。\n\n"
        + build_case_summary(session.to_dict()),
    )
    return session.to_dict()


def build_symptom_text(state: Dict[str, Any]) -> str:
    """将多轮收集的信息转换为现有工作流兼容的一段病例描述。"""
    session = ConsultationSession.from_dict(state)
    patient = session.patient
    parts = []
    age, gender = patient.get("age"), patient.get("gender")
    if age and age != UNKNOWN_VALUE and gender and gender != UNKNOWN_VALUE:
        parts.append(f"患者{gender}，{age}岁")
    elif age and age != UNKNOWN_VALUE:
        parts.append(f"患者{age}岁")
    elif gender and gender != UNKNOWN_VALUE:
        parts.append(f"患者{gender}")

    labels = [
        ("chief_complaint", "主诉"),
        ("duration", "病程"),
        ("accompanying", "伴随症状"),
        ("history", "既往史"),
        ("medications", "正在使用的药物"),
        ("allergy", "过敏史"),
        ("vitals_tests", "生命体征/检查结果"),
        ("extra", "其他补充"),
    ]
    for key, label in labels:
        value = patient.get(key)
        if value and value != UNKNOWN_VALUE:
            parts.append(f"{label}: {value}")
    for update in session.followup_updates:
        if update.strip():
            parts.append(f"后续补充: {update.strip()}")
    return "，".join(parts)


def build_retrieval_query(state: Dict[str, Any]) -> str:
    """构建无“患者/主诉”等格式词的检索查询，沿用现有相关性判定策略。"""
    session = ConsultationSession.from_dict(state)
    patient = session.patient
    keys = ["chief_complaint", "accompanying", "history", "medications", "allergy", "vitals_tests", "extra"]
    parts = [patient.get(key, "") for key in keys]
    parts.extend(session.followup_updates)
    return " ".join(str(value).strip() for value in parts if value and value != UNKNOWN_VALUE)


def build_case_summary(state: Dict[str, Any]) -> str:
    """输出适合直接放入聊天窗口的病例核对摘要。"""
    session = ConsultationSession.from_dict(state)
    patient = session.patient
    rows = [
        ("年龄", patient.get("age", UNKNOWN_VALUE)),
        ("性别", patient.get("gender", UNKNOWN_VALUE)),
        ("主诉", patient.get("chief_complaint", UNKNOWN_VALUE)),
        ("起病/病程", patient.get("duration", UNKNOWN_VALUE)),
        ("伴随症状", patient.get("accompanying", UNKNOWN_VALUE)),
        ("既往史", patient.get("history", UNKNOWN_VALUE)),
        ("正在使用的药物", patient.get("medications", UNKNOWN_VALUE)),
        ("过敏史", patient.get("allergy", UNKNOWN_VALUE)),
        ("生命体征/检查结果", patient.get("vitals_tests", UNKNOWN_VALUE)),
        ("其他补充", patient.get("extra", UNKNOWN_VALUE)),
    ]
    text = "## 已收集病例资料\n\n" + "\n".join(f"- **{label}**：{value or UNKNOWN_VALUE}" for label, value in rows)
    if session.followup_updates:
        text += "\n- **报告后补充**：" + "；".join(session.followup_updates)
    return text


def history_to_chatbot(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """把角色消息转换为 Gradio 6 Chatbot 的 messages 格式，并附上头像名。"""
    session = ConsultationSession.from_dict(state)
    msgs = []
    for m in session.history:
        role = m.get("role", "assistant")
        title = "🤖 机器医生" if role == "assistant" else "👤 你"
        msgs.append({"role": role, "content": m.get("content", ""),
                     "metadata": {"title": title}})
    return msgs
