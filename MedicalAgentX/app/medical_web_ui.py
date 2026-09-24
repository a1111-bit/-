"""
EvoAgentX 医疗智能分析系统 - Web UI（多轮对话问诊版）
基于 Gradio，聊天式引导问诊 → 信息确认 → 完整分析 → 报告后追问
"""

import os
import sys
import threading
import logging

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Windows 控制台默认 GBK 编码无法输出 emoji，强制改用 UTF-8（os.environ 只影响子进程，需 reconfigure 当前进程）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr

import medical_conversation as mc
from evoagentx_medical_workflow import MedicalWorkflowExecutor
from evoagentx_medical_config import OUTPUTS_DIR

# 头像图标（放在 assets/ 目录下；Gradio 的 avatar_images 需要真实文件路径）
_AVATARS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_AVATAR_USER = os.path.join(_AVATARS_DIR, "avatar_user.svg")
# 机器医生头像换成网上下载的卡通机器人医生图（CC0 公共领域，openclipart）
_AVATAR_BOT = os.path.join(_AVATARS_DIR, "avatar_doctor.png")

# 卡通配图（公共领域，离线本地引用；Gradio 里 <img>/CSS 引用本地文件需用 /file= 绝对路径）
def _file_url(filename: str) -> str:
    return os.path.join(_AVATARS_DIR, filename).replace("\\", "/")

_HERO_IMG_URL = _file_url("hero_medical.png")     # 顶部横幅主图：卡通医生
_BG_DOODLE_URL = _file_url("bg_doodle.png")       # 背景装饰图：卡通医疗十字

# 主题切换按钮的内联脚本（放在 <head> 里才会执行；Gradio 的 gr.HTML 走 innerHTML，<script> 不执行）
_HEAD_HTML = """
<script>
(function () {
  var KEY = 'mx-theme';
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  var theme = saved || ((window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', theme);
})();
</script>
"""

_executor_instance = None
_executor_lock = threading.Lock()


def get_executor() -> MedicalWorkflowExecutor:
    global _executor_instance
    if _executor_instance is None:
        with _executor_lock:
            if _executor_instance is None:
                _executor_instance = MedicalWorkflowExecutor()
    return _executor_instance


# ============ 界面辅助：从会话状态计算各组件的显示值 ============

_STAGE_LABELS = {
    mc.STAGE_INTAKE: "📝 问诊进行中：请按提示逐项补充资料，可回答“无 / 不清楚 / 跳过”。",
    mc.STAGE_CONFIRM: "✅ 资料收集完成：请核对下方摘要，确认无误后点击「开始完整分析」，也可继续补充。",
    mc.STAGE_ANALYZING: "🔍 正在分析：正在检索医学文献并执行多智能体推理，请稍候。",
    mc.STAGE_FOLLOWUP: "💬 分析完成：可继续追问，或补充新症状/检查结果后点击「纳入重新分析」。",
    mc.STAGE_EMERGENCY: "🚨 紧急情况：已识别危险信号，请立即线下就医。",
}

# 每个阶段 → CSS class 后缀，让徽章配色走样式表（浅色 + 深色各一套，见 CUSTOM_CSS）
_STAGE_CLASSES = {
    mc.STAGE_INTAKE: "intake",      # 青蓝：收集中
    mc.STAGE_CONFIRM: "confirm",    # 绿：已确认
    mc.STAGE_ANALYZING: "analyzing",# 琥珀：分析中
    mc.STAGE_FOLLOWUP: "followup",  # 青：分析完成
    mc.STAGE_EMERGENCY: "emergency",# 红：紧急
}


def _stage_label(session: dict) -> str:
    """把当前阶段渲染成带颜色的状态徽章 HTML（class 化，深色模式下自动换色）。"""
    stage = session.get("stage", mc.STAGE_INTAKE)
    text = _STAGE_LABELS.get(stage, "")
    key = _STAGE_CLASSES.get(stage, "intake")
    return f'<div class="stage-badge stage-{key}">{text}</div>'


def _loading_html(session: dict) -> str:
    """分析中显示加载动画，其余阶段返回空字符串。"""
    if session.get("stage") == mc.STAGE_ANALYZING:
        return (
            '<div class="loading-box"><span class="spinner"></span>'
            '正在检索医学文献并执行多智能体分析，请稍候…</div>'
        )
    return ""


def _strip_code_fence(text):
    if not text or not isinstance(text, str):
        return ""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1:]
        else:
            s = s.lstrip("`").lstrip("markdown").lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    return s.strip()


def _detail_markdowns(session: dict):
    """从最近一次完整分析结果中取出检索 / 推理 / 工具三块内容。"""
    workflow = (session.get("analysis") or {}).get("workflow_results") or {}

    retriever = workflow.get("retriever_output") or {}
    similar_cases = retriever.get("similar_cases") or ""
    retriever_analysis = retriever.get("retriever_analysis") or ""
    rag_md = ""
    if similar_cases:
        rag_md += "## 📚 检索到的相似医学案例\n\n" + similar_cases + "\n\n---\n\n"
    if retriever_analysis:
        rag_md += "## 🔍 Agent 整理分析\n\n" + retriever_analysis
    if not rag_md:
        rag_md = "_（暂无检索结果）_"

    reasoner = workflow.get("reasoner_output") or {}
    medical_analysis = reasoner.get("medical_analysis") or ""
    primary_diagnoses = reasoner.get("primary_diagnoses") or ""
    recommended_tests = reasoner.get("recommended_tests") or ""
    reasoner_md = ""
    if medical_analysis:
        reasoner_md += "## 🧠 医学推理分析\n\n" + medical_analysis + "\n\n---\n\n"
    if primary_diagnoses:
        reasoner_md += "## 🎯 主要诊断方向\n\n" + primary_diagnoses + "\n\n---\n\n"
    if recommended_tests:
        reasoner_md += "## 🔬 建议检查项目\n\n" + recommended_tests
    if not reasoner_md:
        reasoner_md = "_（暂无推理结果）_"

    tools = workflow.get("tools_output") or {}
    tool_consultation = tools.get("tool_consultation") or ""
    additional_guidance = tools.get("additional_guidance") or ""
    tools_md = ""
    if tool_consultation:
        tools_md += "## 💊 工具咨询结果\n\n" + tool_consultation + "\n\n---\n\n"
    if additional_guidance:
        tools_md += "## 📝 补充指导\n\n" + additional_guidance
    if not tools_md:
        tools_md = "_（暂无工具咨询结果）_"

    return rag_md, reasoner_md, tools_md


def _report_markdown(session: dict) -> str:
    """取出最近一次完整分析生成的综合报告全文（此前生成后未在页面展示）。"""
    analysis = session.get("analysis") or {}
    workflow = analysis.get("workflow_results") or {}
    report = workflow.get("report_output", {}).get("comprehensive_report") or ""
    if not report:
        report = analysis.get("final_report") or ""
    report = _strip_code_fence(report)
    return report or "_（分析完成后此处将显示完整报告）_"


def _render(state: dict):
    """把会话状态渲染为全部 12 个 UI 输出的值。"""
    session = state if isinstance(state, dict) and state else mc.create_session()
    stage = session.get("stage", mc.STAGE_INTAKE)
    rag_md, reasoner_md, tools_md = _detail_markdowns(session)
    report_md = _report_markdown(session)
    return (
        mc.history_to_chatbot(session),                      # 0 chatbot
        gr.update(value=""),                                 # 1 msg（清空输入框）
        session,                                             # 2 state
        _stage_label(session),                               # 3 stage_label
        gr.update(visible=(stage == mc.STAGE_CONFIRM)),      # 4 开始完整分析
        gr.update(visible=(stage == mc.STAGE_FOLLOWUP and bool(session.get("last_followup")))),  # 5 纳入重新分析
        gr.update(visible=True),                             # 6 新建病例/重新开始
        rag_md,                                              # 7 rag_md
        reasoner_md,                                         # 8 reasoner_md
        tools_md,                                            # 9 tools_md
        report_md,                                           # 10 report_md
        _loading_html(session),                              # 11 loading
    )


# ============ 核心流程 ============

def _run_full_analysis(state: dict) -> dict:
    """执行一次完整 4-Agent 工作流，并把结果写回会话状态。"""
    executor = get_executor()
    symptom_text = mc.build_symptom_text(state)
    retrieval_query = mc.build_retrieval_query(state)
    result = executor.execute_medical_analysis(symptom_text, retrieval_query=retrieval_query)
    return mc.record_analysis_result(state, result)


def on_send(state, text):
    text = (text or "").strip()
    if not text:
        return _render(state)

    state = mc.handle_user_message(state, text)

    # 报告后追问：用户在 followup 阶段提问，只走追问接口，不重跑 RAG
    if state.get("stage") == mc.STAGE_FOLLOWUP and state.get("pending_followup"):
        executor = get_executor()
        case_summary = mc.build_case_summary(state)
        previous_report = (state.get("analysis") or {}).get("final_report") or ""
        answer = executor.answer_followup_question(
            case_summary, previous_report, state.get("pending_followup", "")
        )
        state = mc.record_followup_answer(state, answer)

    return _render(state)


def on_analyze(state):
    state = mc.begin_analysis(state)
    if state.get("stage") != mc.STAGE_ANALYZING:
        return _render(state)
    try:
        state = _run_full_analysis(state)
    except Exception as e:
        state = mc.record_analysis_failure(state, str(e))
    return _render(state)


def on_reanalyze(state):
    """把最新追问并入病例，并立即重跑完整工作流。"""
    state = mc.include_last_followup_in_reanalysis(state)
    state = mc.begin_analysis(state)
    if state.get("stage") != mc.STAGE_ANALYZING:
        return _render(state)
    try:
        state = _run_full_analysis(state)
    except Exception as e:
        state = mc.record_analysis_failure(state, str(e))
    return _render(state)


def on_restart():
    return _render(mc.create_session())


CUSTOM_CSS = """
/* ============ 浅色·临床信任主题（语义变量，日夜两套共用） ============ */
:root {
    --body-background-fill: #ecfeff;
    --body-text-color: #164e63;
    --body-text-color-subdued: #475569;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #ffffff;
    --background-fill-tertiary: #ecfeff;
    --border-color-primary: #a5f3fc;
    --border-color-accent: #0891b2;
    --color-accent: #0891b2;
    --color-accent-soft: rgba(8, 145, 178, 0.10);
    --link-text-color: #0891b2;
    --input-background-fill: #ffffff;
    --input-border-color: #a5f3fc;
    --block-background-fill: #ffffff;
    --block-border-color: #e2e8f0;
    --block-label-text-color: #475569;
    --block-title-text-color: #164e63;
    --shadow-drop: 0 4px 16px rgba(8, 145, 178, 0.08);

    /* 语义扩展（补齐硬编码色） */
    --title-gradient: linear-gradient(135deg, #ffffff 0%, #ecfeff 55%, #cffafe 100%);
    --title-border: #a5f3fc;
    --title-shadow: 0 4px 20px rgba(8, 145, 178, 0.10);
    --chat-user-bg: #0891b2;
    --chat-user-fg: #ffffff;
    --chat-user-shadow: 0 2px 8px rgba(8, 145, 178, 0.25);
    --chat-bot-bg: #ffffff;
    --chat-bot-fg: #164e63;
    --chat-bot-border: #e2e8f0;
    --chat-bot-shadow: 0 2px 8px rgba(8, 145, 178, 0.06);
    --tech-badge-bg: rgba(8, 145, 178, 0.10);
    --tech-badge-border: #a5f3fc;
    --tech-badge-fg: #0891b2;
    --primary-gradient: linear-gradient(135deg, #0891b2 0%, #059669 100%);
    --primary-shadow: 0 4px 14px rgba(8, 145, 178, 0.25);
    --primary-shadow-hover: 0 6px 20px rgba(8, 145, 178, 0.35);
    --secondary-bg: #ffffff;
    --secondary-border: #a5f3fc;
    --secondary-fg: #0891b2;
    --secondary-hover-bg: #ecfeff;
    --secondary-hover-border: #0891b2;
    --secondary-hover-fg: #0e7490;
    --avatar-ring: rgba(8, 145, 178, 0.35);
    --loading-bg: rgba(245, 158, 11, 0.12);
    --loading-fg: #b45309;
    --loading-border: rgba(245, 158, 11, 0.35);
    --disclaimer-bg: rgba(245, 158, 11, 0.10);
    --disclaimer-border: #f59e0b;
    --disclaimer-fg: #b45309;
    --focus-ring: rgba(8, 145, 178, 0.4);
    --report-heading: #0891b2;
    --report-rule: #a5f3fc;
    --code-bg: #f1f5f9;
    --table-border: #e2e8f0;
}

/* ============ 深色·夜间模式 ============ */
:root[data-theme="dark"] {
    --body-background-fill: #0b1120;
    --body-text-color: #e2e8f0;
    --body-text-color-subdued: #94a3b8;
    --background-fill-primary: #0f172a;
    --background-fill-secondary: #0f172a;
    --background-fill-tertiary: #0b1120;
    --border-color-primary: #1e293b;
    --border-color-accent: #22d3ee;
    --color-accent: #22d3ee;
    --color-accent-soft: rgba(34, 211, 238, 0.12);
    --link-text-color: #22d3ee;
    --input-background-fill: #0f172a;
    --input-border-color: #334155;
    --block-background-fill: #0f172a;
    --block-border-color: #1e293b;
    --block-label-text-color: #94a3b8;
    --block-title-text-color: #e2e8f0;
    --shadow-drop: 0 4px 16px rgba(0, 0, 0, 0.4);

    --title-gradient: linear-gradient(135deg, #0f172a 0%, #0b1120 60%, #082f49 100%);
    --title-border: #1e293b;
    --title-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    --chat-user-bg: #0891b2;
    --chat-user-fg: #ffffff;
    --chat-user-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    --chat-bot-bg: #0f172a;
    --chat-bot-fg: #e2e8f0;
    --chat-bot-border: #1e293b;
    --chat-bot-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    --tech-badge-bg: rgba(34, 211, 238, 0.12);
    --tech-badge-border: #164e63;
    --tech-badge-fg: #67e8f9;
    --primary-gradient: linear-gradient(135deg, #0891b2 0%, #059669 100%);
    --primary-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    --primary-shadow-hover: 0 6px 20px rgba(0, 0, 0, 0.5);
    --secondary-bg: #0f172a;
    --secondary-border: #334155;
    --secondary-fg: #67e8f9;
    --secondary-hover-bg: #164e63;
    --secondary-hover-border: #22d3ee;
    --secondary-hover-fg: #a5f3fc;
    --avatar-ring: rgba(34, 211, 238, 0.4);
    --loading-bg: rgba(245, 158, 11, 0.16);
    --loading-fg: #fbbf24;
    --loading-border: rgba(245, 158, 11, 0.4);
    --disclaimer-bg: rgba(245, 158, 11, 0.12);
    --disclaimer-border: #f59e0b;
    --disclaimer-fg: #fbbf24;
    --focus-ring: rgba(34, 211, 238, 0.5);
    --report-heading: #67e8f9;
    --report-rule: #164e63;
    --code-bg: #0b1120;
    --table-border: #1e293b;
}

body { background: var(--body-background-fill) !important; }

/* 全页背景装饰卡通图（低透明度，不干扰正文） */
body::before {
    content: "";
    position: fixed; inset: 0;
    background-image: url("/gradio_api/file=__BG_DOODLE__");
    background-size: 460px;
    background-repeat: repeat;
    background-position: center;
    opacity: 0.045;
    pointer-events: none;
    z-index: 0;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto;
    background: transparent;
    font-family: "Figtree", "Noto Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    color: var(--body-text-color);
    position: relative; z-index: 1;
}

/* ============ 顶部横幅：卡通主图 + 标题文字 ============ */
.title-bar {
    display: flex; align-items: center; gap: 24px;
    background: var(--title-gradient);
    color: var(--body-text-color); padding: 32px 24px; border-radius: 20px;
    text-align: left; margin-bottom: 24px;
    box-shadow: var(--title-shadow);
    border: 1px solid var(--title-border);
}
.title-bar .hero-img {
    width: 200px; max-width: 34%; height: auto;
    border-radius: 16px; flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(8, 145, 178, 0.18);
}
.title-bar .title-text { flex: 1; min-width: 0; }
.title-bar h1 { color: var(--body-text-color) !important; margin: 0 0 8px 0; font-size: 30px; letter-spacing: 0.5px; font-weight: 700; line-height: 1.25; }
.title-bar p { color: var(--body-text-color-subdued) !important; margin: 0; font-size: 15px; letter-spacing: 0.5px; line-height: 1.6; }

/* 阶段状态徽章（class 化，浅/深两套） */
.stage-badge {
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.6;
    margin: 8px 0 16px;
    border: 1px solid transparent;
}
.stage-intake    { color: #0e7490; background: rgba(8, 145, 178, 0.10); border-color: rgba(8, 145, 178, 0.35); }
.stage-confirm   { color: #047857; background: rgba(5, 150, 105, 0.14); border-color: rgba(5, 150, 105, 0.35); }
.stage-analyzing { color: #b45309; background: rgba(245, 158, 11, 0.18); border-color: rgba(245, 158, 11, 0.40); }
.stage-followup  { color: #0f766e; background: rgba(20, 184, 166, 0.16); border-color: rgba(20, 184, 166, 0.40); }
.stage-emergency { color: #b91c1c; background: rgba(220, 38, 38, 0.12); border-color: rgba(220, 38, 38, 0.40); }
:root[data-theme="dark"] .stage-intake    { color: #67e8f9; background: rgba(34, 211, 238, 0.12); }
:root[data-theme="dark"] .stage-confirm   { color: #34d399; background: rgba(52, 211, 153, 0.14); }
:root[data-theme="dark"] .stage-analyzing { color: #fbbf24; background: rgba(245, 158, 11, 0.16); }
:root[data-theme="dark"] .stage-followup  { color: #2dd4bf; background: rgba(45, 212, 191, 0.14); }
:root[data-theme="dark"] .stage-emergency { color: #f87171; background: rgba(239, 68, 68, 0.16); }

/* 顶部技术徽章（去掉装饰 emoji） */
.tech-badges { margin-top: 16px; display: flex; gap: 8px; justify-content: flex-start; flex-wrap: wrap; }
.tech-badge {
    background: var(--tech-badge-bg);
    border: 1px solid var(--tech-badge-border);
    color: var(--tech-badge-fg);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* 分析中加载动画 */
.loading-box {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; margin: 8px 0 16px;
    background: var(--loading-bg); color: var(--loading-fg);
    border: 1px solid var(--loading-border); border-radius: 12px;
    font-size: 14px; font-weight: 500;
}
.spinner {
    width: 16px; height: 16px; flex-shrink: 0;
    border: 2px solid var(--loading-border);
    border-top-color: var(--loading-fg);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* 聊天气泡：用户 vs 助手（区分颜色） */
.chat-panel .message.user .message-content {
    background: var(--chat-user-bg);
    color: var(--chat-user-fg);
    border-radius: 16px 16px 4px 16px;
    box-shadow: var(--chat-user-shadow);
}
.chat-panel .message.bot .message-content {
    background: var(--chat-bot-bg);
    border: 1px solid var(--chat-bot-border);
    color: var(--chat-bot-fg);
    border-radius: 16px 16px 16px 4px;
    box-shadow: var(--chat-bot-shadow);
}

/* 头像更圆、更醒目（机器医生 + 用户） */
.chat-panel .avatar-container img {
    width: 40px; height: 40px; border-radius: 50%; object-fit: cover;
    border: 2px solid var(--avatar-ring);
}

.disclaimer {
    background: var(--disclaimer-bg) !important;
    border: 1px dashed var(--disclaimer-border) !important;
    padding: 14px 18px !important;
    border-radius: 12px !important;
    color: var(--disclaimer-fg) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    text-align: center;
    margin-top: 20px !important;
}

footer { display: none !important; }

button { cursor: pointer !important; }

button.btn-primary {
    background: var(--primary-gradient) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 22px !important;
    box-shadow: var(--primary-shadow) !important;
    transition: all 0.2s ease !important;
}
button.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--primary-shadow-hover) !important;
    filter: brightness(1.03);
}

button.secondary {
    background: var(--secondary-bg) !important;
    border-radius: 10px !important;
    border: 1px solid var(--secondary-border) !important;
    color: var(--secondary-fg) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    background: var(--secondary-hover-bg) !important;
    border-color: var(--secondary-hover-border) !important;
    color: var(--secondary-hover-fg) !important;
}

/* 可访问性：键盘焦点清晰可见 */
button:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

/* ============ 报告 / 详情可读性 ============ */
.report-md, .detail-md { line-height: 1.7 !important; font-size: 15px !important; }
.report-md h2, .detail-md h2 {
    border-left: 4px solid var(--report-heading);
    padding-left: 12px; margin: 24px 0 12px;
    color: var(--block-title-text-color);
    font-size: 18px; font-weight: 700;
}
.report-md h3, .detail-md h3 {
    margin: 18px 0 8px; color: var(--report-heading);
    font-size: 16px; font-weight: 600;
}
.report-md ul, .detail-md ul, .report-md ol, .detail-md ol { padding-left: 22px; margin: 8px 0; }
.report-md li, .detail-md li { margin: 4px 0; }
.report-md p, .detail-md p { margin: 8px 0; }
.report-md table, .detail-md table {
    border-collapse: collapse; width: 100%; margin: 12px 0;
    border: 1px solid var(--table-border);
}
.report-md th, .detail-md th, .report-md td, .detail-md td {
    border: 1px solid var(--table-border); padding: 8px 10px; text-align: left;
}
.report-md code, .detail-md code {
    background: var(--code-bg); padding: 2px 6px; border-radius: 6px;
    font-family: "JetBrains Mono", Consolas, monospace; font-size: 13px;
}
.report-md pre, .detail-md pre {
    background: var(--code-bg); padding: 12px; border-radius: 10px; overflow-x: auto;
    border: 1px solid var(--block-border-color);
}
.report-md hr, .detail-md hr { border: none; border-top: 1px solid var(--report-rule); margin: 20px 0; }

/* ============ 日夜切换按钮（右上角悬浮） ============ */
.theme-toggle {
    position: fixed; top: 18px; right: 18px; z-index: 9999;
    background: var(--secondary-bg); color: var(--secondary-fg);
    border: 2px solid #0891b2; border-radius: 999px;
    padding: 8px 14px; font-size: 14px; font-weight: 600;
    box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.18), 0 4px 16px rgba(8, 145, 178, 0.28);
    cursor: pointer;
    transition: all 0.2s ease;
}
.theme-toggle:hover { border-color: #0e7490; }
.theme-toggle .t-sun { display: none; }
:root[data-theme="dark"] .theme-toggle .t-moon { display: none; }
:root[data-theme="dark"] .theme-toggle .t-sun { display: inline; }
/* 夜间模式：边框换成更亮的青 + 更强发光，与白天同款「青边框+发光」风格 */
:root[data-theme="dark"] .theme-toggle {
    background: #1e293b;
    border-color: #22d3ee;
    color: #a5f3fc;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.22), 0 4px 16px rgba(0, 0, 0, 0.5);
}
:root[data-theme="dark"] .theme-toggle:hover { background: #164e63; border-color: #a5f3fc; }

/* 尊重系统「减少动态效果」设置 */
@media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}

/* 响应式：平板 / 大屏 */
@media (min-width: 1024px) {
    .title-bar { padding: 36px 32px; }
    .title-bar .hero-img { width: 240px; }
}
/* 响应式：小屏手机 */
@media (max-width: 640px) {
    .gradio-container { max-width: 100% !important; padding: 0 8px; }
    .title-bar { flex-direction: column; text-align: center; padding: 20px 16px; border-radius: 14px; gap: 16px; }
    .title-bar .hero-img { width: 160px; max-width: 80%; }
    .title-bar h1 { font-size: 22px; }
    .title-bar p { font-size: 14px; }
    .tech-badges { justify-content: center; }
    .stage-badge { font-size: 14px; }
    .theme-toggle { top: 10px; right: 10px; padding: 6px 10px; font-size: 13px; }
}
@media (max-width: 375px) {
    .title-bar .hero-img { width: 120px; }
    .title-bar h1 { font-size: 20px; }
}
"""

# 注入本地背景装饰图路径（避免在 CSS 字符串里拼接绝对路径时与 {} 冲突）
CUSTOM_CSS = CUSTOM_CSS.replace("__BG_DOODLE__", _BG_DOODLE_URL)


def build_ui():
    with gr.Blocks(title="EvoAgentX 医疗智能分析系统") as demo:

        # 日夜切换按钮（右上角悬浮；点击逻辑走内联 onclick + <head> 里的初始主题脚本）
        gr.HTML("""
        <button id="mx-theme-toggle" class="theme-toggle" type="button" aria-label="切换日夜模式" title="切换日夜模式"
            onclick="var r=document.documentElement,t=r.getAttribute('data-theme')==='dark'?'light':'dark';r.setAttribute('data-theme',t);try{localStorage.setItem('mx-theme',t)}catch(e){};">
            <span class="t-moon">🌙 夜间模式</span>
            <span class="t-sun">☀️ 日间模式</span>
        </button>
        """)

        gr.HTML("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet">
        <div class="title-bar">
            <img class="hero-img" src="/gradio_api/file=__HERO_IMG__" alt="卡通医生插图">
            <div class="title-text">
                <h1>EvoAgentX 医疗智能分析系统</h1>
                <p>基于 Agentic RAG 的多智能体协作医疗辅助诊断 · 多轮引导问诊</p>
                <div class="tech-badges">
                    <span class="tech-badge">4-Agent 协作</span>
                    <span class="tech-badge">RAG 医学检索</span>
                    <span class="tech-badge">GLM-4-Flash</span>
                </div>
            </div>
        </div>
        """.replace("__HERO_IMG__", _HERO_IMG_URL))

        initial_session = mc.create_session()
        chatbot = gr.Chatbot(
            label="问诊对话", height=520, elem_classes=["chat-panel"],
            avatar_images=[_AVATAR_USER, _AVATAR_BOT],
            value=mc.history_to_chatbot(initial_session),
        )
        stage_label = gr.HTML(_stage_label(initial_session))
        progress_html = gr.HTML("")

        with gr.Row():
            msg = gr.Textbox(
                label="💬 输入回答或问题", scale=6,
                placeholder="在这里输入你的回答或问题，回车发送…"
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Row():
            analyze_btn = gr.Button("🚀 开始完整分析", variant="primary", visible=False)
            reanalyze_btn = gr.Button("🔄 将此补充纳入重新分析", visible=False)
            restart_btn = gr.Button("🆕 新建病例 / 重新开始", variant="secondary")

        with gr.Tabs():
            with gr.Tab("📄 完整报告"):
                report_md = gr.Markdown("_（分析完成后此处将显示完整报告）_", elem_classes=["report-md"])
            with gr.Tab("📚 检索详情"):
                rag_md = gr.Markdown("_（暂无检索结果）_", elem_classes=["detail-md"])
            with gr.Tab("🧠 推理详情"):
                reasoner_md = gr.Markdown("_（暂无推理结果）_", elem_classes=["detail-md"])
            with gr.Tab("💊 工具详情"):
                tools_md = gr.Markdown("_（暂无工具咨询结果）_", elem_classes=["detail-md"])

        state = gr.State(value=initial_session)

        gr.HTML("""
        <div class="disclaimer">
            ⚠️ 免责声明: 本系统基于 AI 辅助分析，输出仅供医疗专业人员参考，
            不能替代正式医疗诊断。最终诊断需要医师结合完整临床信息做出。
        </div>
        """)

        outputs = [chatbot, msg, state, stage_label,
                   analyze_btn, reanalyze_btn, restart_btn,
                   rag_md, reasoner_md, tools_md, report_md, progress_html]

        send_btn.click(fn=on_send, inputs=[state, msg], outputs=outputs)
        msg.submit(fn=on_send, inputs=[state, msg], outputs=outputs)
        analyze_btn.click(fn=on_analyze, inputs=[state], outputs=outputs)
        reanalyze_btn.click(fn=on_reanalyze, inputs=[state], outputs=outputs)
        restart_btn.click(fn=on_restart, outputs=outputs)

    return demo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("🏥 EvoAgentX 医疗智能分析系统 - Web UI（多轮对话问诊版）")
    print("=" * 60)
    print("📡 服务地址: http://127.0.0.1:7860")
    print("⏹️  按 Ctrl+C 退出")
    print("=" * 60)

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=False,
        head=_HEAD_HTML,
        css=CUSTOM_CSS,
        allowed_paths=[_AVATARS_DIR],
    )
