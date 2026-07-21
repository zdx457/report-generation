"""从 metadata.json 自动生成实体提取关键词和映射关系

用法：
    python generate_metadata_keywords.py

输出：
    - 更新 entity_tracker.py 中的关键词列表
    - 生成 DIAGNOSIS_PATTERNS（诊断关键词）
    - 生成 DIAGNOSIS_TO_BODY_PART（诊断→部位映射）
"""
import json
import os
import re

# 读取 metadata.json
METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_pipeline", "report_template", "metadata.json"
)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ═══════════════════════════════════════════════════════════
# 1. 提取诊断关键词（从"诊断结论"中清理）
# ═══════════════════════════════════════════════════════════

def clean_diagnosis(text: str) -> str:
    """清理诊断文本，提取核心疾病名称"""
    # 移除编号（如 "01、"、"1."、"11、"）
    text = re.sub(r'^\d+[、.\s]+', '', text)
    # 移除术后复查类描述
    text = re.sub(r'术后[复查]*', '', text)
    text = re.sub(r'治疗后[复查]*', '', text)
    text = re.sub(r'内固定术后', '', text)
    text = re.sub(r'复查', '', text)
    # 移除正常/未见异常类
    if any(kw in text for kw in ['正常', '未见异常', '未见明显', '无殊', '阴性']):
        return ""
    # 提取核心疾病词（保留关键医学术语）
    return text.strip()

# 提取所有诊断结论
diagnoses = metadata.get("诊断结论", [])
clean_diagnoses = []
for d in diagnoses:
    cleaned = clean_diagnosis(d)
    if cleaned and len(cleaned) > 1:
        clean_diagnoses.append(cleaned)

# 去重
clean_diagnoses = list(set(clean_diagnoses))
clean_diagnoses.sort(key=len, reverse=True)  # 按长度降序

print(f"提取到 {len(clean_diagnoses)} 个诊断关键词")

# ═══════════════════════════════════════════════════════════
# 2. 提取部位关键词（从"部位"和"检查项目"中提取）
# ═══════════════════════════════════════════════════════════

body_parts = metadata.get("部位", [])
print(f"\n标准部位列表: {body_parts}")

# 从检查项目中提取更细粒度的部位
exam_items = metadata.get("检查项目", [])
# 提取常见部位词
fine_grained_parts = set()
for item in exam_items:
    # 移除扫描方式
    item_clean = re.sub(r'(平扫|增强|平扫\+增强|CTA|MRA|MRCP|MRV|CTU)', '', item)
    # 提取部位词
    for bp in body_parts:
        if bp in item_clean:
            fine_grained_parts.add(bp)

print(f"从检查项目提取到 {len(fine_grained_parts)} 个细粒度部位")

# ═══════════════════════════════════════════════════════════
# 3. 生成诊断→部位映射（基于检查项目中的组合）
# ═══════════════════════════════════════════════════════════

def infer_body_part_from_diagnosis(diagnosis: str, exam_items: list, body_parts: list) -> str:
    """从检查项目中推断诊断对应的部位"""
    for item in exam_items:
        if diagnosis in item:
            # 找到包含该诊断的检查项目，提取部位
            for bp in body_parts:
                if bp in item:
                    return bp
    return ""

# 生成映射（只对明确的疾病生成）
diagnosis_to_body_part = {}
for diag in clean_diagnoses[:500]:  # 限制数量，避免过多
    bp = infer_body_part_from_diagnosis(diag, exam_items, body_parts)
    if bp:
        diagnosis_to_body_part[diag] = bp

print(f"\n生成 {len(diagnosis_to_body_part)} 个诊断→部位映射")

# ═══════════════════════════════════════════════════════════
# 4. 输出 Python 代码
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("生成的 Python 代码片段：")
print("="*60)

# 输出诊断关键词（前100个）
print("\n# 诊断关键词（前100个）")
print("DIAGNOSIS_PATTERNS = [")
for d in clean_diagnoses[:100]:
    print(f'    "{d}",')
print("]")

# 输出诊断→部位映射（前50个）
print("\n# 诊断→部位映射（前50个）")
print("DIAGNOSIS_TO_BODY_PART = {")
for i, (diag, bp) in enumerate(list(diagnosis_to_body_part.items())[:50]):
    print(f'    "{diag}": "{bp}",')
print("}")
