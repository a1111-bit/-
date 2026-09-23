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

# 头像图标（SVG 文件，放在 assets/ 目录下；Gradio 的 avatar_images 需要真实文件路径）
_AVATARS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_AVATAR_USER = os.path.join(_AVATARS_DIR, "avatar_user.svg")
_AVATAR_BOT = os.path.join(_AVATARS_DIR, "avatar_doctor.svg")

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

# 每个阶段的状态徽章配色（前景色 / 背景色），让用户一眼看出当前进展与是否紧急
_STAGE_STYLES = {
    mc.STAGE_INTAKE: ("#93c5fd", "rgba(59, 130, 246, 0.16)"),    # 蓝：收集中
    mc.STAGE_CONFIRM: ("#86efac", "rgba(34, 197, 94, 0.16)"),     # 绿：已确认
    mc.STAGE_ANALYZING: ("#fcd34d", "rgba(245, 158, 11, 0.16)"),  # 黄：分析中
    mc.STAGE_FOLLOWUP: ("#5eead4", "rgba(20, 184, 166, 0.16)"),   # 青：分析完成
    mc.STAGE_EMERGENCY: ("#fca5a5", "rgba(239, 68, 68, 0.16)"),   # 红：紧急
}


def _stage_label(session: dict) -> str:
    """把当前阶段渲染成带颜色的状态徽章 HTML。"""
    stage = session.get("stage", mc.STAGE_INTAKE)
    text = _STAGE_LABELS.get(stage, "")
    fg, bg = _STAGE_STYLES.get(stage, ("#93c5fd", "rgba(59, 130, 246, 0.16)"))
    return (
        f'<div class="stage-badge" style="color:{fg};background:{bg};'
        f'border:1px solid {fg}40;">{text}</div>'
    )


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
/* ============ 深色科技主题 ============ */
:root {
    --body-background-fill: #0f172a;
    --body-text-color: #e2e8f0;
    --body-text-color-subdued: #94a3b8;
    --background-fill-primary: #1e293b;
    --background-fill-secondary: #1e293b;
    --background-fill-tertiary: #0b1220;
    --border-color-primary: #334155;
    --border-color-accent: #38bdf8;
    --color-accent: #38bdf8;
    --color-accent-soft: rgba(56, 189, 248, 0.15);
    --link-text-color: #7dd3fc;
    --input-background-fill: #0b1220;
    --input-border-color: #334155;
    --block-background-fill: #1e293b;
    --block-border-color: #334155;
    --block-label-text-color: #cbd5e1;
    --block-title-text-color: #e2e8f0;
    --shadow-drop: 0 8px 24px rgba(0, 0, 0, 0.5);
}

body { background: #0f172a !important; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto;
    background: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    color: #e2e8f0;
}

.title-bar {
    background: linear-gradient(135deg, #0b1020 0%, #1e3a8a 50%, #0ea5e9 100%);
    color: white; padding: 32px 24px; border-radius: 16px;
    text-align: center; margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(14, 165, 233, 0.25);
    border: 1px solid rgba(125, 211, 252, 0.25);
}
.title-bar h1 { color: #ffffff !important; margin: 0 0 8px 0; font-size: 30px; letter-spacing: 1px; }
.title-bar p { color: #bae6fd !important; margin: 0; font-size: 14px; letter-spacing: 0.5px; }

/* 阶段状态徽章 */
.stage-badge {
    padding: 10px 16px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.6;
    margin: 8px 0 16px;
}

/* 顶部技术徽章 */
.tech-badges { margin-top: 14px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.tech-badge {
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(125, 211, 252, 0.4);
    color: #e0f2fe;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.3px;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.2);
}

/* 分析中加载动画 */
.loading-box {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; margin: 8px 0 16px;
    background: rgba(245, 158, 11, 0.12); color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px;
    font-size: 14px; font-weight: 500;
}
.spinner {
    width: 16px; height: 16px; flex-shrink: 0;
    border: 2px solid rgba(245, 158, 11, 0.3);
    border-top-color: #fcd34d;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* 聊天气泡：用户 vs 助手（区分颜色） */
.chat-panel .message.user .message-content {
    background: #2563eb;
    color: #ffffff;
    border-radius: 14px 14px 4px 14px;
}
.chat-panel .message.bot .message-content {
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    border-radius: 14px 14px 14px 4px;
}

/* 头像更圆、更醒目（机器医生 + 用户） */
.chat-panel .avatar-container img {
    width: 40px; height: 40px; border-radius: 50%; object-fit: cover;
    border: 2px solid rgba(56, 189, 248, 0.35);
}

.disclaimer {
    background: rgba(234, 88, 12, 0.12) !important;
    border: 1px dashed #f97316 !important;
    padding: 14px 18px !important;
    border-radius: 10px !important;
    color: #fdba74 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    text-align: center;
    margin-top: 20px !important;
}

footer { display: none !important; }

button.btn-primary {
    background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 22px !important;
    box-shadow: 0 4px 18px rgba(14, 165, 233, 0.35) !important;
    transition: all 0.25s ease !important;
}
button.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(14, 165, 233, 0.5) !important;
}

button.secondary {
    background: #1e293b !important;
    border-radius: 8px !important;
    border-color: #475569 !important;
    color: #cbd5e1 !important;
}
button.secondary:hover { background: #273449 !important; border-color: #38bdf8 !important; color: #e0f2fe !important; }

/* 可访问性：键盘焦点清晰可见 */
button:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.5) !important;
}

/* 响应式：小屏手机 */
@media (max-width: 640px) {
    .gradio-container { max-width: 100% !important; padding: 0 8px; }
    .title-bar { padding: 20px 16px; }
    .title-bar h1 { font-size: 22px; }
    .stage-badge { font-size: 14px; }
}
"""


def build_ui():
    with gr.Blocks(title="EvoAgentX 医疗智能分析系统") as demo:

        gr.HTML("""
        <div class="title-bar">
            <h1>🏥 EvoAgentX 医疗智能分析系统</h1>
            <p>基于 Agentic RAG 的多智能体协作医疗辅助诊断 · 多轮引导问诊</p>
            <div class="tech-badges">
                <span class="tech-badge">🤖 4-Agent 协作</span>
                <span class="tech-badge">📚 RAG 医学检索</span>
                <span class="tech-badge">⚡ GLM-4-Flash</span>
            </div>
        </div>
        """)

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
                label="", show_label=False, scale=6,
                placeholder="在这里输入你的回答或问题，回车发送…"
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Row():
            analyze_btn = gr.Button("🚀 开始完整分析", variant="primary", visible=False)
            reanalyze_btn = gr.Button("🔄 将此补充纳入重新分析", visible=False)
            restart_btn = gr.Button("🆕 新建病例 / 重新开始", variant="secondary")

        with gr.Tabs():
            with gr.Tab("📄 完整报告"):
                report_md = gr.Markdown("_（分析完成后此处将显示完整报告）_")
            with gr.Tab("📚 检索详情"):
                rag_md = gr.Markdown("_（暂无检索结果）_")
            with gr.Tab("🧠 推理详情"):
                reasoner_md = gr.Markdown("_（暂无推理结果）_")
            with gr.Tab("💊 工具详情"):
                tools_md = gr.Markdown("_（暂无工具咨询结果）_")

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
        css=CUSTOM_CSS
    )
