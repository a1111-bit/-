"""
真正的EvoAgentX医疗智能系统配置
True EvoAgentX Medical Intelligence System Configuration

改造说明：
- LLM 从 OpenAI GPT-4 切换为 智谱清言 GLM-4-Flash（永久免费）
- Embedding 从 OpenAI ada-002 切换为 智谱 embedding-2（1024维）
- 使用智谱官方 zhipuai SDK，不兼容 OpenAI SDK 直调
- 维度从 1536 改为 1024（embedding-2 输出维度）
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# 项目配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "medical_cases"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 确保目录存在
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ============ 智谱清言 GLM 配置（免费 API） ============
ZHIPU_LLM_MODEL = "glm-4-flash"                  # 永久免费的 LLM
ZHIPU_EMBEDDING_MODEL = "embedding-2"             # Embedding 模型（1024维）
ZHIPU_EMBEDDING_DIM = 1024

def load_api_key() -> str:
    """加载智谱 API 密钥"""
    api_key_file = PROJECT_ROOT.parent / "zhipu_api_key.txt"
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key文件未找到: {api_key_file}")

# API Key
ZHIPU_API_KEY = load_api_key()

# ============ LLM 配置 ============
# 使用 dict 形式存储，便于在 workflow 中通过 ZhipuAI 客户端调用
LLM_CONFIG = {
    "model": ZHIPU_LLM_MODEL,
    "api_key": ZHIPU_API_KEY,
    "temperature": 0.7,           # 适度温度避免模型陷入重复循环
    "max_tokens": 4000,             # 增大输出空间避免截断
    "stream": False,               # 关闭流式以便解析
    "output_response": False
}

# ============ Embedding 配置 ============
EMBEDDING_CONFIG = {
    "provider": "zhipu",
    "model": ZHIPU_EMBEDDING_MODEL,
    "api_key": ZHIPU_API_KEY,
    "dimensions": ZHIPU_EMBEDDING_DIM
}

# ============ FAISS 索引配置 ============
FAISS_INDEX_PATH = str(CACHE_DIR / "faiss_index.index")
CHUNKS_DATA_PATH = str(CACHE_DIR / "chunks_data.pkl")

# ============ 检索相关性阈值 ============
# FAISS 用余弦相似度（IndexFlatIP + L2 归一化），score 范围约 [-1, 1]。
# 最高分低于此阈值判定为"检索不到"，避免大模型对无关输入凭空编造诊断。
# 此为初始值，需用下方验证样本标定后确认（若误伤真实症状可适当调低）。
SIMILARITY_THRESHOLD = 0.35

# ============ 报告后追问提示词（不属于 4 个 Agent，仅供报告后普通追问使用） ============
# 追问只基于已有病例资料和已生成报告回答，不重新检索、不重跑 4-Agent 工作流。
FOLLOWUP_PROMPT = """你是医疗辅助分析系统的报告解读助手。请基于已有的病例资料和分析报告，回答患者或医生的追问。

## 已收集病例资料
{case_summary}

## 已生成的完整分析报告
{previous_report}

## 用户的追问
{question}

请严格遵循以下要求：
1. 只依据上面已有的病例资料和分析报告回答，不要引入报告之外的新诊断或新结论。
2. 如果已有资料不足以回答该问题，请明确说明"现有资料无法确定"，并建议补充相应检查或线下就医。
3. 回答简洁、条理清晰、使用中文。
4. 结尾务必附上免责声明：本回答基于 AI 辅助分析，仅供医疗专业人员参考，不能替代正式医疗诊断。

回答："""

# ============ 旧版配置（为兼容 workflow.py 引入保留，但实际不再使用） ============
# 这些字段保留是为了让 evoagentx_medical_workflow.py 的现有 import 不报错
# 实际 LLM 调用会使用新的 LLM_CONFIG dict

# 系统/工作流配置（保持不变）
SYSTEM_CONFIG = {
    "max_execution_time": 300,
    "retry_count": 2,
    "log_level": "INFO",
    "save_intermediate_results": True,
    "medical_disclaimer": True
}

# ============ 医疗工作流任务定义（保持原 prompt 不变） ============
MEDICAL_WORKFLOW_TASKS = [
    {
        "name": "MedicalRetriever",
        "description": "从医学文档库中检索相似病例",
        "inputs": [
            {
                "name": "symptom_text",
                "type": "str",
                "required": True,
                "description": "患者症状描述"
            },
        ],
        "outputs": [
            {
                "name": "similar_cases",
                "type": "str",
                "required": True,
                "description": "检索到的相似病例"
            },
            {
                "name": "retrieval_metadata",
                "type": "str",
                "required": True,
                "description": "检索元数据信息"
            }
        ],
        "prompt": """作为医学文档检索专家，你需要分析患者症状并提供相关的医学案例检索。

患者症状描述：
{symptom_text}

请基于给定的症状描述，提供医学案例检索分析：
1. 症状关键词识别
2. 相关医学领域分析
3. 潜在诊断方向
4. 建议检索策略

以结构化格式输出检索分析。""",
        "parse_mode": "str"
    },
    {
        "name": "MedicalReasoner",
        "description": "基于相似病例进行医学推理分析",
        "inputs": [
            {
                "name": "symptom_text",
                "type": "str",
                "required": True,
                "description": "患者症状描述"
            },
            {
                "name": "similar_cases",
                "type": "str",
                "required": True,
                "description": "检索到的相似病例"
            },
            {
                "name": "retrieval_metadata",
                "type": "str",
                "required": True,
                "description": "检索元数据信息"
            }
        ],
        "outputs": [
            {
                "name": "medical_analysis",
                "type": "str",
                "required": True,
                "description": "医学分析结果"
            },
            {
                "name": "primary_diagnoses",
                "type": "str",
                "required": True,
                "description": "主要诊断方向"
            },
            {
                "name": "recommended_tests",
                "type": "str",
                "required": True,
                "description": "建议检查项目"
            }
        ],
        "prompt": """你是一位资深的临床医生，请基于患者症状和相似病例进行专业的医学分析。

患者症状：
{symptom_text}

相似病例分析：
{similar_cases}

检索元数据：
{retrieval_metadata}

重要：如果上述"相似病例分析"与患者症状不相关、或检索元数据显示没有检索到相关医学资料，请直接回答"未检索到与当前输入相关的医学资料，系统不会基于不足证据给出疾病判断。"，不要编造诊断方向。

请提供详细的医学分析，包括：

## 症状分析
- 主要症状特征和临床意义
- 伴随症状的重要性
- 症状发展模式

## 可能病因
- 最可能的3-5个诊断方向
- 每个诊断的支持证据
- 病理生理机制分析

## 鉴别诊断
- 需要排除的疾病
- 鉴别要点和关键特征

## 检查建议
- 必要的实验室检查
- 影像学检查建议
- 特殊检查项目

## 风险评估
- 病情严重程度
- 可能的并发症
- 紧急程度评估

请提供循证医学的专业分析。""",
        "parse_mode": "str"
    },
    {
        "name": "MedicalToolsConsultant",
        "description": "调用医学工具获取补充信息",
        "inputs": [
            {
                "name": "medical_analysis",
                "type": "str",
                "required": True,
                "description": "医学分析结果"
            },
            {
                "name": "primary_diagnoses",
                "type": "str",
                "required": True,
                "description": "主要诊断方向"
            }
        ],
        "outputs": [
            {
                "name": "tool_consultation",
                "type": "str",
                "required": True,
                "description": "工具咨询结果"
            },
            {
                "name": "additional_guidance",
                "type": "str",
                "required": True,
                "description": "额外指导建议"
            }
        ],
        "prompt": """作为医学知识库专家，请基于分析结果提供补充的医学指导。

医学分析结果：
{medical_analysis}

主要诊断方向：
{primary_diagnoses}

请提供以下方面的专业建议：

## 药物指导
- 相关治疗药物选择
- 用药注意事项
- 可能的药物相互作用

## 诊断标准
- 相关疾病的诊断标准
- 临床指南参考
- 最新研究进展

## 治疗建议
- 标准治疗方案
- 个体化治疗考虑
- 预后评估

## 预防措施
- 疾病预防要点
- 生活方式建议
- 随访计划

请提供基于循证医学的专业建议。""",
        "parse_mode": "str"
    },
    {
        "name": "MedicalReportGenerator",
        "description": "生成综合医学分析报告",
        "inputs": [
            {
                "name": "symptom_text",
                "type": "str",
                "required": True,
                "description": "患者症状描述"
            },
            {
                "name": "similar_cases",
                "type": "str",
                "required": True,
                "description": "相似病例分析"
            },
            {
                "name": "medical_analysis",
                "type": "str",
                "required": True,
                "description": "医学分析结果"
            },
            {
                "name": "primary_diagnoses",
                "type": "str",
                "required": True,
                "description": "主要诊断方向"
            },
            {
                "name": "recommended_tests",
                "type": "str",
                "required": True,
                "description": "建议检查项目"
            },
            {
                "name": "tool_consultation",
                "type": "str",
                "required": True,
                "description": "工具咨询结果"
            },
            {
                "name": "additional_guidance",
                "type": "str",
                "required": True,
                "description": "额外指导建议"
            }
        ],
        "outputs": [
            {
                "name": "comprehensive_report",
                "type": "str",
                "required": True,
                "description": "完整的医学分析报告"
            },
            {
                "name": "executive_summary",
                "type": "str",
                "required": True,
                "description": "执行摘要"
            }
        ],
        "prompt": """作为资深医学报告专家，请整合所有分析信息生成一份完整的医学分析报告。

## 输入信息

### 患者症状
{symptom_text}

### 相似病例分析
{similar_cases}

### 医学分析结果
{medical_analysis}

### 主要诊断方向
{primary_diagnoses}

### 建议检查项目
{recommended_tests}

### 工具咨询结果
{tool_consultation}

### 额外指导建议
{additional_guidance}

---

请生成一份结构完整的医学分析报告：

# 医学综合分析报告

## 执行摘要
- 病例概述
- 主要发现
- 关键建议

## 1. 病例基本信息
- 症状描述
- 临床表现特点

## 2. 相似病例参考
- 检索病例匹配情况
- 相似度分析
- 参考价值评估

## 3. 临床分析
### 3.1 症状分析
### 3.2 可能诊断
### 3.3 鉴别诊断

## 4. 检查建议
### 4.1 必需检查
### 4.2 选择性检查

## 5. 治疗指导
### 5.1 药物治疗
### 5.2 非药物治疗
### 5.3 生活指导

## 6. 风险管理
### 6.1 并发症预防
### 6.2 随访计划

## 7. 专业建议
### 7.1 临床决策支持
### 7.2 进一步咨询建议

## 免责声明
本报告基于AI辅助分析，仅供医疗专业人员参考。最终诊断需要医师结合完整临床信息做出。

使用markdown格式，专业且易读。""",
        "parse_mode": "str"
    }
]

# 医学专科配置
MEDICAL_SPECIALTIES = [
    "内科", "外科", "神经科", "心血管科", "消化科",
    "呼吸科", "内分泌科", "血液科", "肿瘤科", "风湿免疫科"
]
