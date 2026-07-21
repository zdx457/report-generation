"""从 metadata.json 智能生成实体提取配置

策略：
1. 从"检查项目"中提取标准的部位词
2. 从"诊断结论"中提取核心疾病关键词（清理冗长描述）
3. 建立诊断→部位的映射（基于检查项目中的组合模式）
"""
import json
import os
import re
from collections import Counter

METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_pipeline", "report_template", "metadata.json"
)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ═══════════════════════════════════════════════════════════
# 1. 从检查项目中提取部位词
# ═══════════════════════════════════════════════════════════

exam_items = metadata.get("检查项目", [])

# 常见部位词根
BODY_PART_ROOTS = [
    "颅脑", "头颅", "头部", "脑部",
    "胸部", "肺部", "胸腔", "肺",
    "腹部", "肝脏", "肝", "胆囊", "胆", "胰腺", "胰", "脾脏", "脾", "肾脏", "肾",
    "盆腔", "子宫", "卵巢", "前列腺", "膀胱",
    "脊柱", "颈椎", "胸椎", "腰椎", "骶椎",
    "膝关节", "膝", "髋关节", "髋", "肩关节", "肩", "肘关节", "肘", "腕关节", "腕",
    "踝关节", "踝", "足", "手",
    "颈部", "甲状腺",
    "心血管", "心脏", "血管", "冠脉", "主动脉",
    "骨骼", "骨",
    "胃肠", "胃", "肠道", "结肠", "直肠",
    "鼻咽", "咽喉", "口腔",
    "乳腺", "眼眶", "副鼻窦", "鼻骨",
]

# 统计部位出现频率
body_part_freq = Counter()
for item in exam_items:
    for bp in BODY_PART_ROOTS:
        if bp in item:
            body_part_freq[bp] += 1

print("部位词频率统计（前20）：")
for bp, count in body_part_freq.most_common(20):
    print(f"  {bp}: {count}")

# ═══════════════════════════════════════════════════════════
# 2. 从诊断结论中提取核心疾病词
# ═══════════════════════════════════════════════════════════

diagnoses = metadata.get("诊断结论", [])

# 常见疾病词根（医学标准术语）
DISEASE_ROOTS = [
    # 脑部疾病
    "脑梗", "脑出血", "脑梗塞", "脑血栓", "脑肿瘤", "脑膜瘤", "脑胶质瘤",
    "脑转移瘤", "脑脓肿", "脑炎", "脑积水", "脑萎缩", "脑挫裂伤",
    # 肺部疾病
    "肺炎", "肺结核", "肺气肿", "肺癌", "肺结节", "肺栓塞", "肺脓肿",
    "支气管肺炎", "大叶性肺炎", "间质性肺炎",
    # 肝脏疾病
    "肝硬化", "肝炎", "肝癌", "肝囊肿", "肝血管瘤", "肝脓肿", "肝转移瘤",
    # 骨科疾病
    "骨折", "骨肿瘤", "骨质疏松", "骨髓炎", "骨转移瘤", "关节炎", "椎间盘突出",
    # 心血管疾病
    "动脉瘤", "主动脉夹层", "冠心病", "心肌梗死", "心肌病",
    # 肿瘤通用词
    "肿瘤", "癌症", "癌", "肉瘤", "淋巴瘤", "转移瘤",
    # 其他常见疾病
    "囊肿", "息肉", "结节", "炎症", "感染", "结石",
    "腹水", "积液", "积水",
]

# 统计疾病词频率
disease_freq = Counter()
for diag in diagnoses:
    for disease in DISEASE_ROOTS:
        if disease in diag:
            disease_freq[disease] += 1

print("\n疾病词频率统计（前30）：")
for disease, count in disease_freq.most_common(30):
    print(f"  {disease}: {count}")

# ═══════════════════════════════════════════════════════════
# 3. 建立诊断→部位映射
# ═══════════════════════════════════════════════════════════

# 分析检查项目中的模式：疾病名通常出现在项目描述的末尾
# 例如："肝硬化+食道下段静脉曲张" → 部位：腹部/肝脏

diagnosis_to_body_part = {}

# 方法1：基于疾病词根直接映射
disease_to_part_mapping = {
    # 脑部
    "脑梗": "脑部", "脑出血": "脑部", "脑梗塞": "脑部", "脑血栓": "脑部",
    "脑肿瘤": "脑部", "脑膜瘤": "脑部", "脑胶质瘤": "脑部", "脑转移瘤": "脑部",
    "脑脓肿": "脑部", "脑炎": "脑部", "脑积水": "脑部", "脑萎缩": "脑部",
    "脑挫裂伤": "脑部",
    # 肺部
    "肺炎": "肺部", "肺结核": "肺部", "肺气肿": "肺部", "肺癌": "肺部",
    "肺结节": "肺部", "肺栓塞": "肺部", "肺脓肿": "肺部",
    "支气管肺炎": "肺部", "大叶性肺炎": "肺部", "间质性肺炎": "肺部",
    # 肝脏
    "肝硬化": "肝脏", "肝炎": "肝脏", "肝癌": "肝脏", "肝囊肿": "肝脏",
    "肝血管瘤": "肝脏", "肝脓肿": "肝脏", "肝转移瘤": "肝脏",
    # 骨骼（不明确，需要上下文）
    "骨折": None, "骨肿瘤": None, "骨质疏松": "骨骼", "骨髓炎": None,
    "骨转移瘤": None, "关节炎": None, "椎间盘突出": "脊柱",
    # 心血管
    "动脉瘤": "血管", "主动脉夹层": "血管", "冠心病": "心脏",
    "心肌梗死": "心脏", "心肌病": "心脏",
    # 通用（不明确）
    "肿瘤": None, "癌症": None, "癌": None, "肉瘤": None,
    "淋巴瘤": None, "转移瘤": None, "囊肿": None, "息肉": None,
    "结节": None, "炎症": None, "感染": None, "结石": None,
    "腹水": "腹部", "积液": None, "积水": None,
}

diagnosis_to_body_part = {k: v for k, v in disease_to_part_mapping.items() if v}

print(f"\n生成 {len(diagnosis_to_body_part)} 个诊断→部位映射")

# ═══════════════════════════════════════════════════════════
# 4. 输出 Python 代码
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("生成的 Python 代码：")
print("="*60)

# 输出部位关键词（按频率排序）
print("\n# 部位关键词（从 metadata 提取）")
print("BODY_PART_PATTERNS = [")
for bp, _ in body_part_freq.most_common():
    print(f'    "{bp}",')
print("]")

# 输出诊断关键词（按频率排序，前80个）
print("\n# 诊断关键词（从 metadata 提取，前80个）")
print("DIAGNOSIS_PATTERNS = [")
for disease, _ in disease_freq.most_common(80):
    print(f'    "{disease}",')
print("]")

# 输出诊断→部位映射
print("\n# 诊断→部位映射（从 metadata 推断）")
print("DIAGNOSIS_TO_BODY_PART = {")
for disease, bp in diagnosis_to_body_part.items():
    print(f'    "{disease}": "{bp}",')
print("}")
