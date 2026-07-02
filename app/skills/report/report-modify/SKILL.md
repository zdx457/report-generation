---
name: report-modify
description: Use when modifying content in an existing medical imaging report. Direct modification path without RAG retrieval — takes the current report content and user's modification requirements, outputs the revised report. Use when user asks to edit/update/revise/change an already generated report.
version: 1.0.0
author: report-generation-team
license: MIT
metadata:
  hermes:
    tags: [医疗影像, 报告修改, 直接修改]
    related_skills: [report-generation]
---

# 修改报告

## Overview

本 Skill 用于修改已有医疗影像报告中的内容，不经过 RAG 检索。用户已生成了一份报告，现需根据要求修改报告内容。采用直接修改模式，LLM 接收当前报告内容和修改要求，输出修改后的完整报告。

## When to Use

- 当用户已生成报告，需要对报告内容进行修改时
- 当用户提出"修改XX部分"、"把XX改成XX"、"更新诊断结论"等修改需求时
- 当用户需要对报告中的描述、诊断结论、影像学表现等内容进行增删改时
- **Don't use for**: 从零生成新报告 → 使用 [report-generation] Skill

## 核心流程

### 1. 接收输入

接收两个关键输入：

1. **当前报告内容**：用户已生成的完整报告（Markdown 格式）
2. **修改要求**：用户对报告的修改需求描述

### 2. System Prompt

```
你是一个医疗影像报告修改助手。用户已经生成了一份报告，现在需要你根据用户的要求修改报告内容。

### 工作方式

1. 用户会提供"当前报告内容"和"修改要求"
2. 你需要根据修改要求，对报告进行精确修改
3. 只修改用户要求修改的部分，其他内容保持不变
4. 直接输出修改后的完整报告

### 重要规则

- 只修改用户明确要求修改的部分，不要擅自改动其他内容
- 保持报告的整体格式和结构不变
- 如果修改要求不明确，请向用户确认
- 输出格式为 Markdown
```

### 3. 组装消息

```python
messages = [
    {"role": "system", "content": MODIFY_SYSTEM_PROMPT},
    {"role": "user", "content": f"当前报告内容：\n{current_report}\n\n修改要求：{modification_request}"}
]

# 调用 LLM 生成修改后的报告
modified_report = chat(messages)
```

### 4. 输出修改后的报告

LLM 直接输出修改后的完整报告（Markdown 格式），代码端无需额外解析。

**成功标准**：

- 修改后的报告仅改动用户指定的部分
- 报告整体格式和结构保持不变
- 未指定的内容与原始报告一致

## Common Pitfalls

1. **LLM 擅自修改未指定的内容**：修改后的报告中，用户未要求改动的部分发生了变化 → 原因是 System Prompt 约束不足，或模型倾向于"优化"报告 → 解决方案：在 System Prompt 中明确强调"只修改用户明确要求修改的部分，不要擅自改动其他内容"，必要时在代码中对比原始报告和修改后的差异。

2. **修改要求不明确**：用户输入 "改一下诊断" 等模糊要求 → 原因是用户未提供具体的修改内容 → 解决方案：在 System Prompt 中指示模型"如果修改要求不明确，请向用户确认"，而非猜测修改意图。

3. **报告格式丢失**：修改后报告的 Markdown 格式（标题层级、列表、表格等）与原始报告不一致 → 原因是 LLM 重新组织了报告结构 → 解决方案：在 System Prompt 中强调"保持报告的整体格式和结构不变"。

4. **修改后报告内容截断**：输出不完整，报告尾部缺失 → 原因是 LLM 输出 token 限制不足 → 解决方案：确保 `max_tokens` 设置足够大（建议 ≥4096），或使用流式输出逐步拼接。

## Verification Checklist

- [ ] 修改要求是否被精确执行，未擅自改动其他内容？
- [ ] 报告整体格式和结构是否与修改前一致？
- [ ] 修改后的报告是否完整（无截断）？
- [ ] 修改要求不明确时，是否向用户确认而非猜测？
- [ ] 输出是否为 Markdown 格式？
- [ ] 修改后的报告内容是否与用户要求一致？
