# Memory 记忆模块

## 概述

为多轮对话系统提供三层分层记忆能力，包括实体追踪、对话历史管理、上下文消解、用户偏好统计和修正历史。模块独立，不依赖任何特定框架或业务领域。

## 三层记忆架构

```
┌──────────────────────────────────────────────────────────┐
│  EntityTracker (实体追踪层) —— "上下文大脑"               │
│  职责：维护当前会话的结构化状态（modality/body_part/...） │
│  特性：继承性 + 覆盖性 + 意图检测                         │
├──────────────────────────────────────────────────────────┤
│  ShortTermMemory (短期记忆层) —— "对话草稿"               │
│  职责：维护当前会话的对话历史和摘要                       │
│  特性：滑动窗口 + 结构化压缩                               │
├──────────────────────────────────────────────────────────┤
│  LongTermMemory (长期记忆层) —— "习惯本"                  │
│  职责：跨会话持久化用户偏好和术语习惯                      │
│  特性：自动注入 System Prompt + 指数衰减                   │
└──────────────────────────────────────────────────────────┘
```

## 设计决策

| 决策项   | 选择              | 说明                                                                   |
| -------- | ----------------- | ---------------------------------------------------------------------- |
| 短期存储 | 内存 dict         | 简单快速，线程安全，重启丢失                                           |
| 长期存储 | SQLite 数据库     | 持久化到 `data/ltm.db`，原子事务，无文件锁竞争，跨会话，重启不丢失     |
| 历史长度 | 最近 6 轮         | 超出后 FIFO 淘汰最早轮次，自动压缩为摘要                               |
| 实体提取 | 规则 + LLM 兜底   | Phase 1: 关键词匹配；Phase 2: LLM JSON 结构化提取                      |
| 框架无关 | 独立模块          | 不依赖任何 Web 框架，通过 `session_id` 和 `user_id` 隔离，适配任意前端 |
| 会话标识 | session_id 字符串 | 由调用方传入，每个会话唯一                                             |
| 用户隔离 | user_id 字符串    | EntityTracker + STM 按 session_id 隔离，LTM 按 user_id 物理隔离        |

## 隔离策略

```
                      EntityTracker + STM (内存)          LTM (磁盘)
                      ┌─────────────────────┐         ┌──────────────────┐
用户A 会话1 ─────────▶│ session_abc         │         │                  │
用户A 会话2 ─────────▶│ session_def         │──偏好─▶│   data/ltm.db    │
用户B 会话1 ─────────▶│ session_ghi         │  同步  │  (按 user_id)    │
用户C 会话1 ─────────▶│ session_jkl         │────────▶│                  │
                      └─────────────────────┘         └──────────────────┘
```

| 维度     | EntityTracker + ShortTermMemory | LongTermMemory                            |
| -------- | ------------------------------- | ----------------------------------------- |
| 隔离键   | `session_id`（会话级别）        | `user_id`（用户级别）                     |
| 隔离粒度 | 会话级，互不可见                | 用户级，同一用户多会话共享                |
| 物理存储 | 内存，进程重启丢失              | 磁盘 `data/ltm.db`，SQLite 原子事务持久化 |

## 模块结构

```
app/memory/
├── __init__.py          # 包入口，导出 EntityTracker / ShortTermMemory / LongTermMemory
├── entity_tracker.py    # 实体追踪器（独立类，槽位模型）
├── short_term.py        # 短期记忆（对话历史 + 摘要压缩）
├── long_term.py         # 长期记忆（用户偏好统计 + 修正历史 + 持久化）
├── chat_test.py         # 多轮对话测试终端
├── data/                # 长期记忆持久化目录
│   └── ltm.db           # SQLite 数据库（偏好、修正历史、统计）
└── README.md            # 本文件
```

## 核心功能

### 1. EntityTracker —— "上下文大脑"

负责维护当前对话的结构化状态，解决"用户在说什么"的问题。

#### 槽位模型

```python
slots = {
    "modality": None,          # 当前模态 (CT, MR, DR...)
    "body_part": None,         # 当前部位 (Brain, Chest, Liver...)
    "clinical_history": "",    # 病史/症状
    "diagnosis": [],           # 已确认的诊断列表
    "intent": "new_session",   # 当前意图: new_session / append / switch
}
```

#### 核心特性

- **继承性**：用户说"再看看肝脏"，自动继承上一轮的模态 CT
- **覆盖性**：用户说"换成 MR 膝关节"，清空旧状态，建立新状态
- **意图检测**：识别 `new_session` / `append` / `switch` 三种意图
- **切换清洗**：`switch` 意图时强制清空 STM 和 last_report，严禁旧病灶残留

#### 实体提取策略

- **Phase 1（当前）**：规则匹配（关键词列表，按长度降序匹配）
- **Phase 2（计划中）**：LLM JSON 结构化提取（`{"modality": "CT", "body_part": "肝脏"}`）

#### API

```python
from memory import EntityTracker

tracker = EntityTracker()
```

| 方法                         | 说明                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `update_from_query(query)`   | 从用户输入提取实体，更新槽位，返回变更 dict                       |
| `detect_intent(query)`       | 检测意图：`new_session` / `append` / `switch`                     |
| `resolve_context(query)`     | 上下文消解：补全省略信息（如"再看看肝脏" → "CT 肝脏 再看看肝脏"） |
| `apply_switch(query)`        | 切换意图：清空所有槽位，从新查询重新提取                          |
| `clear()`                    | 重置所有槽位到初始状态                                            |
| `to_dict()`                  | 序列化为字典                                                      |
| `to_context_prompt()`        | 生成上下文提示片段，用于注入 System Prompt                        |
| `set_clinical_history(text)` | 设置病史                                                          |
| `add_diagnosis(text)`        | 添加诊断                                                          |

### 2. ShortTermMemory —— "对话草稿"

负责维护当前会话的对话历史和摘要，解决"刚才说了什么"的问题。

#### 核心特性

- **滑动窗口**：默认保留最近 6 轮完整交互
- **结构化压缩**：淘汰旧轮次时调用 LLM 压缩为摘要，保留关键实体和临床信息
- **纯对话管理**：实体追踪已拆分至 EntityTracker，职责单一

#### 摘要机制

- 优先使用 LLM 摘要（`summarize_fn`），失败回退到规则截断
- 摘要存储在 `_summaries[session_id]` 列表中
- `build_messages()` 自动将摘要作为额外的 system 消息注入

#### API

```python
from memory import ShortTermMemory

stm = ShortTermMemory(max_rounds=6, summarize_fn=my_llm_summarize)
```

| 方法                                           | 说明                                                      |
| ---------------------------------------------- | --------------------------------------------------------- |
| `add_turn(sid, user_msg, assistant_msg)`       | 记录一轮对话，超 max_rounds 时自动触发摘要                |
| `get_history(sid)`                             | 获取对话历史（list[dict]）                                |
| `get_last_turn(sid)`                           | 获取最近一轮对话                                          |
| `get_summaries(sid)`                           | 获取已淘汰轮次的摘要列表                                  |
| `build_messages(sid, system_prompt, user_msg)` | 组装完整 messages 列表（System + 摘要 + 历史 + 当前输入） |
| `clear(sid)`                                   | 清除指定会话的记忆（含摘要）                              |
| `active_sessions()`                            | 当前活跃会话数                                            |
| `session_info(sid)`                            | 获取会话统计（轮次、摘要数）                              |
| `cleanup_expired(max_age_seconds=3600)`        | 清理超时会话                                              |

### 3. LongTermMemory —— "习惯本"

负责跨会话的用户偏好和术语习惯，解决"用户喜欢什么"的问题。

#### 核心特性

- **自动注入**：新会话开始时，偏好自动注入 System Prompt 顶部
- **指数衰减**：根据使用频率和时间衰减权重，遗忘过时偏好
- **写缓存**：内存 `_cache` 读写，后台定时器异步刷新到 SQLite
- **会话结束强制落盘**：`on_session_end()` 取消定时器立即刷新

#### 偏好计分

`Score = Σ e^(-λ·Δt)`，λ 由半衰期决定（默认 7 天）。较新的偏好权重高，旧的随时间衰减。

#### API

```python
from memory import LongTermMemory

ltm = LongTermMemory(user_id="user_001", half_life_days=7.0, max_age_days=30.0, flush_interval=300.0)
```

| 方法                                            | 说明                                        |
| ----------------------------------------------- | ------------------------------------------- |
| `update_preferences(entities)`                  | 追加偏好记录（带时间戳）                    |
| `get_preferences()`                             | 获取时间衰减加权后的偏好（top5 + 加权得分） |
| `get_preference_prompt()`                       | 生成偏好提示文本，用于注入 System Prompt    |
| `add_correction(question, original, corrected)` | 记录一条修正历史（few-shot 示例）           |
| `get_corrections(limit=5)`                      | 获取最近 N 条修正历史                       |
| `get_correction_prompt()`                       | 生成修正参考提示                            |
| `sync_from_short_term(stm, session_id)`         | 从短期记忆同步实体到长期记忆                |
| `on_session_end(stm, session_id)`               | 会话结束时更新统计并保存                    |
| `get_stats()`                                   | 获取统计信息（会话数、总轮次）              |
| `clear()`                                       | 清空当前用户的长期记忆                      |
| `close()`                                       | 取消定时器 + 强制落盘                       |

#### 数据库 (`data/ltm.db`)

```sql
preferences(user_id, key, value, timestamp)      -- 偏好记录
corrections(user_id, id, question, original, corrected, timestamp)  -- 修正历史
stats(user_id, total_sessions, total_turns, last_updated)           -- 统计
```

## 流水线集成

在 `run_pipeline` 中，记忆模块在以下三个阶段介入：

### 阶段 1：输入处理 (Pre-LLM)

```
1. 实体提取：entity_tracker.update_from_query(query)
2. 意图检测：entity_tracker.detect_intent(query)
   → 若 switch：清空 STM + last_report + entity_tracker（彻底清洗）
3. 上下文消解：entity_tracker.resolve_context(query)
```

### 阶段 2：Prompt 构建 (Context Injection)

```
[System Prompt 顶部]
{ltm.get_preference_prompt()}        ← LTM 偏好（优先级最高）
{entity_tracker.to_context_prompt()} ← 当前实体上下文
{STRUCTURE_PROMPT / EDIT_PROMPT}     ← 任务专用 Prompt
```

### 阶段 3：输出生成后 (Post-LLM)

```
1. stm.add_turn(session_id, query, response)
2. ltm.sync_from_short_term(stm, session_id)（会话结束时）
```

## 集成示例

```python
from memory import EntityTracker, ShortTermMemory, LongTermMemory

entity_tracker = EntityTracker()
stm = ShortTermMemory(max_rounds=6, summarize_fn=my_summarize_fn)
ltm = LongTermMemory(user_id="user_001")

def respond(query, session_id):
    # ── 阶段 1：输入处理 ──
    entity_tracker.update_from_query(query)

    if entity_tracker.detect_intent(query) == "switch":
        stm.clear(session_id)
        last_report = ""
        entity_tracker.apply_switch(query)

    enhanced = entity_tracker.resolve_context(query)

    # ── 阶段 2：Prompt 构建 ──
    sys_prompt = STRUCTURE_PROMPT
    pref = ltm.get_preference_prompt()
    ctx = entity_tracker.to_context_prompt()
    if pref:
        sys_prompt = pref + "\n\n" + sys_prompt
    if ctx:
        sys_prompt = sys_prompt + "\n\n" + ctx

    # ── 调用 LLM ──
    messages = [{"role": "system", "content": sys_prompt}]
    history = stm.get_history(session_id)
    messages.extend(history)
    messages.append({"role": "user", "content": enhanced})
    response = call_llm(messages)

    # ── 阶段 3：输出处理后 ──
    stm.add_turn(session_id, query, response)
    return response

def clear_session(session_id):
    stm.clear(session_id)
    entity_tracker.clear()
```

## 测试

### 多轮对话测试

```bash
cd app/memory
python chat_test.py                # 普通模式
python chat_test.py --debug        # 调试模式（显示上下文消解和记忆状态）
```

### CLI 测试

```bash
cd app
python -m chat.rag_chat_v2         # 三段式工作流 CLI
```

### 内置命令

| 命令            | 说明                                                |
| --------------- | --------------------------------------------------- |
| `exit` / `quit` | 退出（自动触发 `on_session_end`，更新长期记忆统计） |
| `clear`         | 清空当前会话：STM + EntityTracker + last_report     |
| `info`          | 查看短期记忆状态 + 实体槽位 + 长期记忆统计          |
| `ltminfo`       | 查看长期记忆详情（偏好、统计、偏好提示文本）        |

### 测试场景

1. **实体继承**：先输入"CT 头部"，再输入"再看看肝脏" → 模态继承 CT，部位更新为肝脏
2. **意图切换清洗**：先输入"CT 脑部 脑出血"生成报告，再输入"换成 MR 膝关节" → 旧报告清空，实体槽位重置
3. **上下文消解**：输入"再看看这个" → 自动补全模态和部位
4. **LTM 偏好注入**：检查 `structure_report` 调用时 System Prompt 中是否包含偏好提示
5. **智能摘要**：连续 6 轮以上，检查淘汰的旧轮次是否生成了摘要
6. **长期记忆持久化**：退出后重启，检查 `ltminfo` 是否保留了之前的会话统计
