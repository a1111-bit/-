"""
医疗智能分析工作流（基于智谱清言 GLM 直调，不依赖 EvoAgentX 工作流引擎）

改造说明：
- 原 workflow.py 依赖 EvoAgentX 的 SequentialWorkFlowGraph/WorkFlow/AgentManager/OpenAILLM
- 但 EvoAgentX 的 OpenAILLM 通过 litellm 代理，无法识别智谱的模型名
- 这里重写工作流，直接通过 ZhipuAI SDK 调用 GLM-4-Flash，逻辑等价但更简洁
- 4 个 Agent 顺序执行（检索→推理→工具咨询→报告生成），保持原 prompt 不变
- 对外接口（execute_medical_analysis / get_workflow_info）与原版一致
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from zhipuai import ZhipuAI

from evoagentx_medical_config import (
    LLM_CONFIG, MEDICAL_WORKFLOW_TASKS, SYSTEM_CONFIG,
    OUTPUTS_DIR, PROJECT_ROOT,
    ZHIPU_API_KEY, ZHIPU_LLM_MODEL,
    FOLLOWUP_PROMPT
)
from evoagentx_medical_engine import MedicalRAGEngine
from tooluniverse_integration import MedicalToolUniverseWrapper


# 输入相关性判定提示词：在正式分析前拦截与医学无关的输入（问候/闲聊/乱码等）
RELEVANCE_GATE_PROMPT = (
    "请判断下面这段文本是否属于医学症状或疾病的描述。\n"
    "判断标准：提到了症状（如头痛、发热、咳嗽）、疾病、检查、用药等医学内容才算相关。\n"
    "如果只是问候、闲聊、天气、乱码、纯数字等与医学无关的内容，算无关。\n"
    "请只回答两个字：相关 或 无关。\n\n"
    "文本：{text}"
)


class MedicalWorkflowExecutor:
    """医疗工作流执行器（智谱 GLM 版）"""

    def __init__(self):
        """初始化医疗工作流执行器"""
        self.logger = logging.getLogger(__name__)

        # 初始化医疗RAG引擎
        self.medical_rag = MedicalRAGEngine()

        # 初始化ToolUniverse医学工具
        self.tool_universe = MedicalToolUniverseWrapper()

        # 初始化智谱 LLM 客户端
        self.llm_client = ZhipuAI(api_key=ZHIPU_API_KEY)
        self.llm_model = ZHIPU_LLM_MODEL
        self.temperature = LLM_CONFIG.get("temperature", 0.7)
        self.max_tokens = LLM_CONFIG.get("max_tokens", 4000)

        # 工作流图（保留引用以便 get_workflow_info 兼容）
        self.workflow_graph = type("Graph", (), {"goal": "执行基于AI的医疗病例分析，提供诊断支持和治疗建议"})()
        self.agents_count = len(MEDICAL_WORKFLOW_TASKS)

        self.logger.info("医疗工作流执行器初始化完成（智谱 GLM）")

    # ============ LLM 调用 ============

    def _call_llm(self, prompt_template: str, inputs: Dict[str, Any]) -> str:
        """调用智谱 GLM LLM，把 inputs 填入 prompt 模板"""
        # 简单的 .format() 模板填充
        try:
            prompt = prompt_template.format(**inputs)
        except KeyError as e:
            # 找不到字段时，用空字符串替代，避免崩溃
            self.logger.warning(f"prompt 模板字段缺失: {e}")
            safe_inputs = {k: v for k, v in inputs.items()}
            for k in ["symptom_text", "similar_cases", "retrieval_metadata",
                      "medical_analysis", "primary_diagnoses", "recommended_tests",
                      "tool_consultation", "additional_guidance"]:
                safe_inputs.setdefault(k, "")
            prompt = prompt_template.format(**safe_inputs)

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一位资深的临床医学专家，请提供专业、严谨、循证医学的医学分析。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            self.logger.error(f"智谱 LLM 调用失败: {e}")
            return f"[LLM 调用失败: {e}]"

    # ============ 索引 ============

    def ensure_rag_indexed(self, force_reindex: bool = False) -> bool:
        """确保RAG引擎已建立索引"""
        try:
            if not self.medical_rag.is_indexed or force_reindex:
                self.logger.info("正在建立医学文档索引...")
                success = self.medical_rag.index_medical_documents(force_reindex)
                if not success:
                    self.logger.error("建立医学文档索引失败")
                    return False
                self.logger.info("医学文档索引建立完成")
            return True
        except Exception as e:
            self.logger.error(f"建立索引时发生错误: {str(e)}")
            return False

    # ============ 主流程 ============

    def execute_medical_analysis(self, symptom_text: str, retrieval_query: str = None, **kwargs) -> Dict[str, Any]:
        """执行医疗分析工作流（4个 Agent 顺序协作）"""
        try:
            # 1. 确保RAG索引存在
            if not self.ensure_rag_indexed():
                return {
                    "status": "error",
                    "error": "无法建立医学文档索引",
                    "timestamp": datetime.now().isoformat()
                }

            self.logger.info(f"开始执行医疗分析: {symptom_text}")

            top_k = kwargs.get("top_k", 5)

            # 用于向量检索的查询文本：网页会传干净的原始症状，避免"患者/主诉"标签抬高相似度
            query = retrieval_query or symptom_text

            # 2. Agent1: 医学检索（RAG + LLM 整理）
            self.logger.info("步骤1/4: 医学文档检索与整理")
            rag_results = self.medical_rag.search_similar_cases(query, top_k=top_k)
            if rag_results["status"] != "success":
                return {
                    "status": "error",
                    "error": f"文档检索失败: {rag_results.get('error', '未知错误')}",
                    "timestamp": datetime.now().isoformat()
                }

            retrieval_metadata = json.dumps(rag_results["search_metadata"], ensure_ascii=False)

            # 相关性判断（第一道：相似度门槛）——检索不到相关内容时直接返回，不让模型编造诊断
            if not rag_results.get("relevant", True) or rag_results.get("total_results", 0) == 0:
                self.logger.info("检索结果与症状无关，判定为检索不到，跳过推理")
                return self._no_relevant_evidence_result(symptom_text, rag_results, retrieval_metadata)

            # 相关性判断（第二道：LLM 判定）——拦截"你好"这类相似度虚高但医学无关的短词
            if not self._is_medical_relevant(query):
                self.logger.info("LLM 判定输入与医学无关，返回检索不到")
                return self._no_relevant_evidence_result(symptom_text, rag_results, retrieval_metadata)

            similar_cases = self._format_rag_results(rag_results)

            retriever_prompt = self._get_task_prompt("MedicalRetriever")
            retriever_output = self._call_llm(retriever_prompt, {
                "symptom_text": symptom_text
            })

            # 3. Agent2: 医学推理
            self.logger.info("步骤2/4: 执行医学推理分析")
            reasoner_prompt = self._get_task_prompt("MedicalReasoner")
            medical_analysis = self._call_llm(reasoner_prompt, {
                "symptom_text": symptom_text,
                "similar_cases": similar_cases,
                "retrieval_metadata": retrieval_metadata
            })
            primary_diagnoses = self._extract_diagnoses(medical_analysis)
            recommended_tests = self._extract_tests(medical_analysis)

            # 4. Agent3: 工具咨询（ToolUniverse + LLM）
            self.logger.info("步骤3/4: 执行医学工具咨询")
            diagnoses_list = self._parse_diagnoses_list(primary_diagnoses)
            tooluniverse_results = self.tool_universe.comprehensive_medical_search(
                medical_analysis, diagnoses_list
            ) if self.tool_universe.tool_engine else None

            tools_prompt = self._get_task_prompt("MedicalToolsConsultant")
            tool_consultation = self._call_llm(tools_prompt, {
                "medical_analysis": medical_analysis,
                "primary_diagnoses": primary_diagnoses
            })
            additional_guidance = self._extract_guidance(tool_consultation)

            # 5. Agent4: 报告生成
            self.logger.info("步骤4/4: 生成综合医学报告")
            report_prompt = self._get_task_prompt("MedicalReportGenerator")
            comprehensive_report = self._call_llm(report_prompt, {
                "symptom_text": symptom_text,
                "similar_cases": similar_cases,
                "medical_analysis": medical_analysis,
                "primary_diagnoses": primary_diagnoses,
                "recommended_tests": recommended_tests,
                "tool_consultation": tool_consultation,
                "additional_guidance": additional_guidance
            })
            executive_summary = self._extract_summary(comprehensive_report)

            workflow_results = {
                "retriever_output": {
                    "similar_cases": similar_cases,
                    "retrieval_metadata": retrieval_metadata,
                    "retriever_analysis": retriever_output
                },
                "reasoner_output": {
                    "medical_analysis": medical_analysis,
                    "primary_diagnoses": primary_diagnoses,
                    "recommended_tests": recommended_tests
                },
                "tools_output": {
                    "tool_consultation": tool_consultation,
                    "additional_guidance": additional_guidance,
                    "tooluniverse_results": tooluniverse_results,
                    "real_medical_tools": self.tool_universe.get_tool_status()
                },
                "report_output": {
                    "comprehensive_report": comprehensive_report,
                    "executive_summary": executive_summary
                },
                "comprehensive_report": comprehensive_report,
                "executive_summary": executive_summary
            }

            # 6. 保存结果
            self._save_workflow_results(workflow_results, symptom_text)
            self.logger.info("医疗分析工作流执行完成")

            return {
                "status": "success",
                "input": symptom_text,
                "timestamp": datetime.now().isoformat(),
                "rag_results": rag_results,
                "workflow_results": workflow_results,
                "final_report": comprehensive_report,
                "executive_summary": executive_summary
            }
        except Exception as e:
            self.logger.error(f"执行医疗分析时发生错误: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # ============ 辅助方法 ============

    def _get_task_prompt(self, task_name: str) -> str:
        """从配置获取指定 task 的 prompt"""
        for task in MEDICAL_WORKFLOW_TASKS:
            if task["name"] == task_name:
                return task["prompt"]
        return ""

    def _is_medical_relevant(self, text: str) -> bool:
        """用 LLM 判断输入是否与医学相关，拦截问候/闲聊等无关内容。"""
        try:
            answer = self._call_llm(RELEVANCE_GATE_PROMPT, {"text": text}).strip()
            if "无关" in answer:
                return False
            return True  # 相关或无法判断时放行，避免误伤真实症状
        except Exception:
            return True

    def _no_relevant_evidence_result(self, symptom_text: str, rag_results: Dict[str, Any],
                                     retrieval_metadata: str) -> Dict[str, Any]:
        """构造"检索不到"的统一返回结构（不执行任何 LLM 推理/工具/报告）。"""
        msg = "未检索到与当前输入相关的医学资料，系统不会基于不足证据给出疾病判断。"
        workflow_results = {
            "retriever_output": {"similar_cases": "未检索到相关医学案例。",
                                 "retrieval_metadata": retrieval_metadata,
                                 "retriever_analysis": ""},
            "reasoner_output": {"medical_analysis": "", "primary_diagnoses": "",
                                "recommended_tests": ""},
            "tools_output": {"tool_consultation": "", "additional_guidance": ""},
            "report_output": {"comprehensive_report": msg, "executive_summary": msg},
            "comprehensive_report": msg,
            "executive_summary": msg
        }
        self._save_workflow_results(workflow_results, symptom_text)
        self.logger.info("已返回检索不到提示，未执行推理/工具/报告")
        return {
            "status": "success",                      # 外层保持 success，兼容 CLI / 网页
            "analysis_status": "no_relevant_evidence",
            "low_relevance": True,
            "input": symptom_text,
            "timestamp": datetime.now().isoformat(),
            "rag_results": rag_results,
            "workflow_results": workflow_results,
            "final_report": msg,
            "executive_summary": msg
        }

    def answer_followup_question(self, case_summary: str, previous_report: str, question: str) -> str:
        """报告后的普通追问：只基于已有病例资料和报告回答，不重跑 RAG 与 4-Agent 工作流。

        供多轮对话网页在 followup 阶段调用；命令行/批处理的单次完整分析不经过这里。
        """
        try:
            answer = self._call_llm(FOLLOWUP_PROMPT, {
                "case_summary": case_summary or "未提供",
                "previous_report": previous_report or "未生成完整报告",
                "question": question or "",
            })
            return answer.strip() or "暂时无法生成回答，请稍后重试或咨询线下医生。"
        except Exception as e:
            self.logger.error(f"报告后追问失败: {e}")
            return "暂时无法生成回答，请稍后重试或咨询线下医生。"

    def _format_rag_results(self, rag_results: Dict[str, Any]) -> str:
        """格式化RAG检索结果（截断内容避免上下文过长导致模型退化）"""
        if not rag_results.get("results"):
            return "未找到相关医学案例。"
        formatted = "检索到的相关医学案例:\n\n"
        for r in rag_results["results"]:
            # 限制每个 chunk 内容长度，避免上下文过长触发模型退化
            content = r["content"]
            if len(content) > 400:
                content = content[:400] + "..."
            formatted += f"案例 {r['rank']} (相似度: {r['score']:.4f})\n"
            formatted += f"来源: {r['document_title']}\n"
            formatted += f"内容: {content}\n"
            formatted += "-" * 50 + "\n\n"
        return formatted

    def _extract_diagnoses(self, content: str) -> str:
        """从分析内容中提取诊断信息"""
        lines = content.split('\n')
        diagnoses = []
        in_diag = False
        for line in lines:
            line = line.strip()
            if "可能病因" in line or "诊断" in line:
                in_diag = True
                continue
            elif in_diag and line and not line.startswith('#'):
                if any(k in line for k in ['检查', '治疗', '风险']):
                    break
                diagnoses.append(line)
        return "\n".join(diagnoses) if diagnoses else "需要进一步分析确定诊断方向"

    def _extract_tests(self, content: str) -> str:
        """提取检查建议"""
        lines = content.split('\n')
        tests = []
        in_tests = False
        for line in lines:
            line = line.strip()
            if "检查建议" in line or "检查" in line:
                in_tests = True
                continue
            elif in_tests and line and not line.startswith('#'):
                if any(k in line for k in ['治疗', '风险', '随访']):
                    break
                tests.append(line)
        return "\n".join(tests) if tests else "建议常规实验室检查和影像学检查"

    def _extract_guidance(self, content: str) -> str:
        """提取指导建议（截断）"""
        return content[:500] + "..." if len(content) > 500 else content

    def _parse_diagnoses_list(self, diagnoses_text: str) -> List[str]:
        """从诊断文本中解析出诊断列表"""
        diagnoses = []
        for line in diagnoses_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                clean = line.lstrip('0123456789.-、 ')
                if clean and len(clean) > 2:
                    diagnoses.append(clean)
        return diagnoses[:5]

    def _extract_summary(self, content: str) -> str:
        """提取执行摘要"""
        lines = content.split('\n')
        summary_lines = []
        in_summary = False
        for line in lines:
            line = line.strip()
            if "执行摘要" in line:
                in_summary = True
                continue
            elif in_summary:
                if line.startswith('#') and "执行摘要" not in line:
                    break
                if line and not line.startswith('#'):
                    summary_lines.append(line)
        return "\n".join(summary_lines) if summary_lines else "请查看完整报告获取详细信息"

    def _save_workflow_results(self, results: Dict[str, Any], query: str):
        """保存工作流执行结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evoagentx_medical_analysis_{timestamp}.json"
            filepath = OUTPUTS_DIR / "results" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            save_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "executor_type": "ZhipuAI-GLM",
                "workflow_config": {
                    "goal": self.workflow_graph.goal,
                    "tasks_count": len(MEDICAL_WORKFLOW_TASKS),
                    "llm_model": self.llm_model
                },
                "results": results
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"工作流结果已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存工作流结果时出错: {str(e)}")

    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        return {
            "executor_type": "ZhipuAI-GLM",
            "workflow_goal": self.workflow_graph.goal,
            "tasks_count": len(MEDICAL_WORKFLOW_TASKS),
            "agents_count": self.agents_count,
            "llm_config": {
                "model": self.llm_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "provider": "zhipu"
            },
            "rag_engine_info": self.medical_rag.get_corpus_info(),
            "tooluniverse_status": self.tool_universe.get_tool_status(),
            "system_config": SYSTEM_CONFIG
        }


def main():
    """测试医疗工作流"""
    print("🏥 初始化医疗工作流执行器...")
    try:
        executor = MedicalWorkflowExecutor()
        print("\n📊 工作流信息:")
        print(json.dumps(executor.get_workflow_info(), ensure_ascii=False, indent=2))

        test_cases = [
            "患者男性，58岁，近期出现体重下降，皮肤黄疸，上腹部胀痛，ALT和胆红素升高",
            "女性患者，35岁，反复头痛伴恶心呕吐，视物模糊，血压180/110mmHg"
        ]

        for i, case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"🧪 测试案例 {i}: {case}")
            print('='*60)
            results = executor.execute_medical_analysis(case)
            if results["status"] == "success":
                print("✅ 分析成功完成")
                print(f"\n📋 执行摘要:")
                print(results.get("executive_summary", "无摘要"))
                print(f"\n🩺 完整报告:")
                print(results.get("final_report", "无报告"))
            else:
                print(f"❌ 分析失败: {results.get('error', '未知错误')}")

        print(f"\n🎉 医疗工作流测试完成!")
    except Exception as e:
        print(f"❌ 执行过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()
