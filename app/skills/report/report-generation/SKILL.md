---
name: report-generation
description: Use when generating a new medical imaging report from scratch. Performs full RAG pipeline: fuzzy detection → standardization → query rewriting → ReAct multi-step reasoning with multi-recall + rerank retrieval. Use when user asks to generate/create a new report or search for imaging report templates.
version: 1.0.0
author: report-generation-team
license: MIT
metadata:
  hermes:
    tags: [医疗影像, 报告生成, RAG, ReAct, 多路召回]
    related_skills: [report-modify]
---

# 生成报告

## Overview

本 Skill 走完整 RAG 管道生成新的医疗影像报告。核心流程：先对用户输入进行模糊检测和术语标准化，再通过查询改写提升召回精度，最后进入 ReAct 多轮推理循环，结合多路召回（向量检索 + 元数据过滤 + 关键词检索）与 Rerank 精排，生成最终报告。

## When to Use

- 当用户需要生成一份新的医疗影像报告时
- 当用户输入检查类型、部位、诊断描述等关键词，需要检索相关报告模板时
- 当用户需要根据已有报告模板生成符合格式的影像报告时
- **Don't use for**: 修改已有报告的内容 → 使用 [report-modify] Skill

## 核心流程

### 1. 模糊检测

收到用户输入后，首先判断查询是否过于模糊（仅有检查类型，缺少部位和诊断）。

> 执行方式：调用 query_rewrite.is_too_vague 工具进行校验。若返回 True，则终止流程并返回追问提示。

```python
from query_rewrite import is_too_vague, get_clarification

if is_too_vague(user_input):
    clarification = get_clarification(user_input)
    # 返回追问提示，引导用户补充信息
    return clarification
```

**触发条件**：

- 输入仅包含检查类型，如 "CT"、"MR"、"DR"
- 输入无检查类型、无部位、无诊断

**成功标准**：模糊查询被拦截，返回追问提示，不进入后续流程。

### 2. 术语标准化

将用户输入中的非标准术语映射为标准术语。

> 执行方式：调用 query_rewrite.standardize_query 工具。用返回结果替换原查询。

```python
from query_rewrite import standardize_query

query = standardize_query(user_input)
# "CT头部" → "CT头颅"
# "MR颈部" → "MR头颈部"
```

**映射规则**（通过 `PART_ALIASES` 字典维护）：

- "头"/"脑"/"颅脑"/"头部" → "头颅"
- "颈"/"颈部" → "头颈部"
- "胸"/"胸廓" → "胸部"
- "腹" → "腹部"
- "脊椎" → "脊柱"

### 3. 查询改写

如果查询有检查类型+部位但缺少诊断信息，调用 LLM 将查询扩展为更具体的描述。

> 执行方式：调用 query_rewrite.rewrite_query 工具。若改写结果与原查询不同，则使用改写后的查询进入后续步骤。

```python
from query_rewrite import needs_rewrite, rewrite_query

if needs_rewrite(query):
    rewritten = rewrite_query(query)
    if rewritten != query:
        query = rewritten
```

**成功标准**：改写后的查询包含更丰富的诊断描述，提升后续多路召回的精准度。

### 4. ReAct 多轮推理循环

进入 ReAct 推理循环，最大步数 `MAX_STEPS = 5`。

#### 4.1 System Prompt

````
你是一个具备多步推理能力的 AI 助手，你有访问报告数据库的检索工具。

## 输出格式

每轮你必须输出以下三种格式之一：

### 继续推理
```
[CONTINUE]
你对当前问题的推理分析（可以是一段话，也可以是多点分析）
```

### 调用检索
```
[ACTION: search]
检索查询词（简洁的搜索关键词）
```

### 最终回答
```
[FINAL]
你的最终回答（Markdown 格式，基于检索结果和推理给出准确回答）
```

## 工作方式

1. 收到问题后，先判断是否需要检索：
   - 需要检索 → 输出 [ACTION: search] 进行检索
   - 不需要检索 → 输出 [CONTINUE] 进行推理
2. 检索结果会以"观察"形式返回给你
3. 综合分析检索结果和推理，判断是否需要继续：
   - 信息不足 → 继续 [ACTION: search] 或 [CONTINUE]
   - 信息充分 → 输出 [FINAL] 给出最终回答

## 重要规则

- 第一轮不要直接输出 [FINAL]，至少先检索或推理一步
- 检索时使用简洁的关键词，不要用完整句子
- [FINAL] 回答要基于检索结果，注明信息来源（如"参考1显示..."）
- 如果检索结果不足以回答问题，如实说明
````

#### 4.2 解析输出

```python
from rag_chat import parse_react_output

action_type, payload = parse_react_output(llm_output)
# action_type: "final" | "action" | "continue"
# 当 action_type == "action" 时，payload 为 (action_name, action_input)
# 例如: ("search", "CT头颅脑出血")
```

#### 4.3 执行检索

当 LLM 输出 `[ACTION: search]` 时，执行多路召回 + Rerank：

> 执行方式：依次调用 retrieval.multi_recall 和 rerank.rerank_documents 工具。注意设置 top_k=5 和 top_n=3。

```python
from retrieval import multi_recall
from rerank import rerank_documents

# 步骤1: 生成查询向量
query_vec = get_embedding(query)

# 步骤2: 解析关键词
keywords = parse_query_keywords(query)

# 步骤3: 多路召回（向量检索 + 元数据过滤 + 关键词检索）
candidates = multi_recall(query_vec, keywords, top_k=5, client=client)

# 步骤4: Rerank 精排
documents = [e["text"] for e in candidates]
rerank_results = rerank_documents(query, documents, top_n=3)
```

**检索参数**：

- 向量检索 Top-K: 5
- Rerank 精排 Top-K: 3
- 最大推理步数: 5

**三路召回说明**：

1. **向量检索（Bi-Encoder）**：基于 bge-m3 向量的语义相似度匹配，适合模糊查询
2. **元数据过滤**：按检查类型/部位/诊断结论精确匹配，使用 Milvus filter 表达式
3. **关键词检索**：基于 Milvus like 查询的全文匹配，适合专业术语检索

**去重逻辑**：三路召回结果按 `(source, text)` 合并去重，记录 `_recall_paths` 标记命中路径。

### 5. 最终回答

当 LLM 输出 `[FINAL]` 时，提取最终回答并返回。回答格式为 Markdown，需注明信息来源。

## Common Pitfalls

1. **LLM 第一轮直接输出 [FINAL]**：模型跳过检索直接给出最终回答 → 原因是 System Prompt 未生效或模型推理过短 → 解决方案：在 System Prompt 中强调"第一轮不要直接输出 [FINAL]，至少先检索或推理一步"，并在代码中校验第一轮输出。

2. **检索结果为空**：多路召回返回 0 条结果 → 原因可能是查询关键词与数据库内容不匹配，或向量数据库未正确加载 → 解决方案：检查 `MilvusClient.load_collection()` 是否调用成功，检查 `COLLECTION_NAME` 是否匹配，必要时提示用户修改查询。

3. **Rerank API 调用失败**：SiliconFlow Rerank API 超时或返回错误 → 原因可能是 API Key 未配置或网络问题 → 解决方案：代码中已实现 fallback 逻辑，Rerank 失败时降级为直接使用 Top-K 向量检索结果。

4. **查询过于模糊导致检索不精准**：用户输入 "CT" 未被拦截 → 原因可能是 `is_too_vague()` 判断逻辑不完善 → 解决方案：检查 `KNOWN_TYPES` 和 `KNOWN_PARTS` 是否与 `metadata.json` 同步，必要时调用 `reload_metadata()` 刷新。

5. **术语标准化不生效**：用户输入 "CT头部" 未被标准化为 "CT头颅" → 原因可能是 `PART_ALIASES` 缺少对应映射 → 解决方案：在 `PART_ALIASES` 字典中补充映射关系。

## Verification Checklist

- [ ] 模糊查询（如 "CT"、"MR"）是否被正确拦截并返回追问提示？
- [ ] 术语标准化是否正确映射（如 "头部" → "头颅"、"颈部" → "头颈部"）？
- [ ] 查询改写是否生成了更丰富的诊断描述？
- [ ] ReAct 循环是否在 MAX_STEPS=5 步内完成？
- [ ] 多路召回是否返回了去重后的候选文档？
- [ ] Rerank 精排是否按相关性分数降序返回？
- [ ] 最终回答是否包含信息来源标注（如"参考1显示..."）？
- [ ] 检索结果不足时，是否如实说明而非编造内容？
- [ ] 向量数据库是否已正确加载（`load_collection` 成功）？
- [ ] 环境变量（EMBED_URL, CHAT_URL, RERANK_URL, SILICONFLOW_API_KEY）是否正确配置？
