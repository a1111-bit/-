---
name: medical-agentx-unittest
description: MedicalAgentX 项目的单元测试编写与运行指南。当用户想"跑单元测试""写单元测试""测工作流文本解析""测 test_*.py""验证检索过滤/会话状态机"时使用。
---

# MedicalAgentX 单元测试

本技能指导在 MedicalAgentX（医学 Agentic RAG）项目里**运行**和**新增**单元测试。
核心原则：**只测纯函数**（不联网、不调模型、不建索引），所以跑得飞快、稳定可重复。

## 何时使用

- 用户说"跑一下测试""做单元测试""写单元测试""测测这块逻辑"。
- 用户提到某个 `test_*.py`、某个解析/过滤/状态机函数想验证。
- 改完 `medical_conversation.py`、`evoagentx_medical_workflow.py`、`evoagentx_medical_engine.py`
  后想确认没改坏。

## 现有测试清单

全部位于 `MedicalAgentX/app/` 目录：

| 文件 | 测什么 | 是否联网 |
|---|---|---|
| `test_medical_conversation.py` | 会话状态机：问诊顺序、跳过值、年龄校验、危险信号、确认/重分析 | 否（纯本地） |
| `test_rag_filter.py` | 检索相关性过滤 `filter_relevant_results` | 否 |
| `test_workflow_helpers.py` | 工作流 6 个文本解析纯方法（诊断/检查/摘要/指导/诊断列表/RAG 格式化） | 否 |

## 如何运行

```bash
# 在项目根目录（MedicalAgentX-Agentic-RAG-main）下执行，任选其一：

# 1) 指定三个文件逐个跑：
cd MedicalAgentX/app && python -m unittest test_medical_conversation test_rag_filter test_workflow_helpers -v

# 2) 一键跑全部 test_*.py：
cd MedicalAgentX/app && python -m unittest discover -p "test_*.py" -v
```

看到 `OK` 或结尾 `Ran N tests ... OK` 即全部通过。

## 如何新增测试

复用现有模式（`test_rag_filter.py` / `test_workflow_helpers.py`）：

1. **文件头**固定三行，保证从任意目录都能 import 到同目录模块：

   ```python
   import sys
   import unittest
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parent))
   ```

2. **只测纯函数**：优先测不依赖 `self`、不联网的字符串/字典处理函数。

3. **测 `MedicalWorkflowExecutor` 的方法时，务必用 `object.__new__` 绕过重 `__init__`**：

   ```python
   from evoagentx_medical_workflow import MedicalWorkflowExecutor
   def _bare_executor():
       return object.__new__(MedicalWorkflowExecutor)  # 跳过 RAG/工具/LLM 初始化
   ```

   因为 `__init__` 会初始化 RAG 引擎、ToolUniverse、智谱客户端（重、需联网）。这些纯解析方法
   不用 `self`，空壳实例完全够用。

4. **写清每个方法的三要素**：触发词 / 停止词 / 空结果兜底。工作流 6 个方法如下：

   | 方法 | 触发 | 停止 | 兜底 |
   |---|---|---|---|
   | `_extract_diagnoses` | 行含「可能病因/诊断」 | 行含「检查/治疗/风险」 | `需要进一步分析确定诊断方向` |
   | `_extract_tests` | 行含「检查」 | 行含「治疗/风险/随访」 | `建议常规实验室检查和影像学检查` |
   | `_extract_summary` | 行含「执行摘要」 | 下一个 `#` 标题行 | `请查看完整报告获取详细信息` |
   | `_extract_guidance` | 无（>500 字截断加 `...`） | — | 原样返回 |
   | `_parse_diagnoses_list` | 去行首序号，过滤 ≤2 字，最多 5 条 | — | 空列表 |
   | `_format_rag_results` | 无结果 | — | `未找到相关医学案例。` |

   注意：停止词只作用于**非 `#` 开头**的行（`not line.startswith('#')` 先过滤），所以测试输入里
   停止行别写成 `## 检查建议`，要写成 `检查建议`。

5. 末尾加 `if __name__ == "__main__": unittest.main(verbosity=2)`，支持 `python test_xxx.py` 直跑。

## 常见坑

- **Windows 控制台 GBK 编码**：测试/验证脚本里 `print` emoji（如 ✅❌📋）可能抛
  `UnicodeEncodeError`。脚本开头加 `sys.stdout.reconfigure(encoding='utf-8')` 即可。
- **`test_workflow_helpers.py` 依赖 API Key 文件**：`import evoagentx_medical_workflow` 会触发
  `evoagentx_medical_config.load_api_key()` 读项目根目录 `zhipu_api_key.txt`。文件缺失时 import 阶段
  就报 `FileNotFoundError`（还没到测试逻辑）。跑测试前确认该文件存在。
- **别在测试里实例化 `MedicalRAGEngine` / `MedicalToolUniverseWrapper` / `ZhipuAI`**——那会联网/建索引，
  把单元测试变成慢且脆的集成测试。
