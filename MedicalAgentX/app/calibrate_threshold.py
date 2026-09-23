"""
阈值标定脚本：用现有索引实测"相关症状"与"无关文本"的最高相似度。
用途：确认 SIMILARITY_THRESHOLD 是否能把两类输入清楚分开。

运行方式（在 app 目录下）：
    python calibrate_threshold.py

注意：每条样本会调用一次智谱 embedding（消耗少量 API 额度）。
"""

import sys
from pathlib import Path

# Windows 控制台 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evoagentx_medical_engine import MedicalRAGEngine
from evoagentx_medical_config import SIMILARITY_THRESHOLD


# 知识库覆盖的症状（应有较高相似度）
RELEVANT_SAMPLES = [
    "患者头痛伴视物模糊，血压180/110mmHg",
    "发热、咳嗽、咽痛、流涕、鼻塞",
    "多饮多尿多食，体重下降，血糖升高",
    "胸闷气短，活动后呼吸困难，长期吸烟",
    "高温环境下剧烈运动后意识障碍、体温升高",
]

# 与疾病无关的文本（应较低相似度）
IRRELEVANT_SAMPLES = [
    "今天天气不错，我想出去玩",
    "我喜欢吃苹果和香蕉",
    "明天下午三点开会",
    "这是一段测试文本，没有医学含义",
    "今天股市大涨，赚了很多钱",
]


def main():
    print(f"当前阈值 SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}\n")

    engine = MedicalRAGEngine()
    if not engine._load_existing_index():
        print("❌ 未找到现有索引，请先运行 --reindex")
        return 1

    print("=" * 70)
    print("【相关症状】最高相似度（应明显偏高）")
    print("=" * 70)
    for q in RELEVANT_SAMPLES:
        r = engine.search_similar_cases(q, top_k=5)
        print(f"  {r['max_score']:.4f}  |  {q}")

    print()
    print("=" * 70)
    print("【无关文本】最高相似度（应明显偏低）")
    print("=" * 70)
    for q in IRRELEVANT_SAMPLES:
        r = engine.search_similar_cases(q, top_k=5)
        print(f"  {r['max_score']:.4f}  |  {q}")

    print()
    print("判断依据：若两类分数有明显分界，且分界点接近当前阈值，则保留；")
    print("否则应调整 config.py 里的 SIMILARITY_THRESHOLD。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
