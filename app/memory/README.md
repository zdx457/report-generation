# Memory 记忆模块

## 概述

为多轮对话系统提供记忆能力，包括对话历史管理、实体追踪、上下文消解、用户偏好统计和修正历史。模块独立，不依赖任何特定框架或业务领域。

## 设计决策

| 决策项   | 选择              | 说明                                                                           |
| -------- | ----------------- | ------------------------------------------------------------------------------ |
| 短期存储 | 内存 dict         | 简单快速，线程安全，重启丢失                                                   |
| 长期存储 | SQLite 数据库     | 持久化到 `data/ltm.db`，原子事务，无文件锁竞争，跨会话，重启不丢失             |
| 历史长度 | 最近 5 轮         | 超出后 FIFO 淘汰最早轮次                                                       |
| 实现范围 | 短期 + 长期记忆   | 短期：对话历史 + 实体追踪 + 上下文消解；长期：偏好统计 + 修正历史              |
| 框架无关 | 独立模块          | 不依赖任何 Web 框架，通过 `session_id` 和 `user_id` 隔离，适配任意前端         |
| 会话标识 | session_id 字符串 | 由调用方传入，每个会话唯一                                                     |
| 用户隔离 | user_id 字符串    | STM 按 session_id 隔离，LTM 按 user_id 物理隔离，`data/ltm.db` 按 user_id 分表 |

## 隔离策略

短期记忆和长期记忆采用不同粒度的隔离：

```
                      STM (内存)                    LTM (磁盘)
                    ┌─────────────┐              ┌──────────────────┐
用户A 会话1 ───────▶│ session_abc │              │                  │
用户A 会话2 ───────▶│ session_def │──偏好同步──▶│                  │
用户B 会话1 ───────▶│ session_ghi │              │   data/ltm.db    │
                    └─────────────┘              │  (按 user_id 分表)│
                    ┌─────────────┐              │                  │
用户C 会话1 ───────▶│ session_jkl │──偏好同步──▶│                  │
                    └─────────────┘              └──────────────────┘
```

| 维度     | ShortTermMemory          | LongTermMemory                            |
| -------- | ------------------------ | ----------------------------------------- |
| 隔离键   | `session_id`（会话级别） | `user_id`（用户级别）                     |
| 隔离粒度 | 会话级，互不可见         | 用户级，同一用户多会话共享                |
| 物理存储 | 内存 dict，进程重启丢失  | 磁盘 `data/ltm.db`，SQLite 原子事务持久化 |

## 模块结构

```
app/memory/
├── __init__.py          # 包入口，导出 ShortTermMemory / LongTermMemory
├── short_term.py        # 短期记忆（对话历史 + 实体追踪 + 上下文消解）
├── long_term.py         # 长期记忆（用户偏好统计 + 修正历史 + 持久化）
├── chat_test.py         # 多轮对话测试终端
├── data/                # 长期记忆持久化目录
│   └── ltm.db           # SQLite 数据库（偏好、修正历史、统计）
└── README.md            # 本文件
```

## 核心功能

### 短期记忆（ShortTermMemory）

#### 1. 对话历史管理

- 按 `session_id` 隔离存储每个用户的对话轮次
- 每轮保存 `user` 和 `assistant` 消息
- 超过 `max_rounds`（默认5）**不直接丢弃**，而是自动压缩为摘要句
- 线程安全（`threading.Lock`）

#### 2. 智能摘要

- 淘汰旧轮次时，调用 `summarize_fn`（LLM）或规则回退，将对话压缩为一句摘要
- LLM 摘要提示：保留关键实体、核心结论、数值信息，忽略寒暄
- 规则回退：`用户: {前80字} | AI: {前80字}`
- 摘要存储在 `_summaries[session_id]` 列表中，随会话生命周期管理
- `build_messages()` 自动将摘要作为额外的 system 消息注入，确保长对话核心信息不丢失

#### 3. 实体追踪

- 实体由调用方传入，记忆模块不依赖向量库字段名
- `add_turn()` 接受可选的 `entities` 参数：`{"category": "电子产品", "brand": "Apple"}`
- `add_entities()` 单独追加实体，不绑定对话轮次
- 实体按轮次独立存储（每轮一条记录），支持跨轮次累积和置信度衰减

#### 4. 上下文消解（槽位填充 + 置信度衰减）

- 不再仅靠正则匹配指代词，改用**槽位填充**机制
- 内部维护逐轮实体记录，每条记录附带 `_round` 轮次标记
- 当用户查询触发指代/追问时，对每个槽位取最近一轮的非空值，按 `0.9^(当前轮-实体轮)` 计算置信度权重
- 仅填充查询中**未出现**的槽位值，避免重复注入
- 示例：
  - 第1轮：`推荐一款Apple手机` → 实体 `{category:手机, brand:Apple, price:5000+}`
  - 第2轮：`那它的电池怎么样` → 检测到"它"指代，填充 `category:手机(1.00) | brand:Apple(1.00)`
  - 第3轮：`有黑色吗` → 检测到追问，填充 `category:手机(0.81) | brand:Apple(0.81)`（置信度衰减）

#### 5. 消息构建

- `build_messages()` 方法：将 system_prompt + 历史摘要 + 历史轮次 + 当前用户消息组装为 LLM 可用的 messages 列表
- 摘要作为额外的 system 消息注入在历史轮次之前，格式为 `## 历史对话摘要（已淘汰轮次的压缩）`
- 直接用于 `chat_stream()` 调用

### 长期记忆（LongTermMemory）

#### 1. 用户偏好统计（时间衰减加权 + 写缓存）

- 接受任意 `{key: value}` 或 `{key: [list]}` 格式的实体，每次记录附时间戳
- 使用**指数衰减加权**：`Score = Σ e^(-λ·Δt)`，λ 由半衰期决定（默认 7 天）
- 较新的偏好权重高，旧的随时间衰减，用户兴趣转移时自动"遗忘"旧偏好
- **内存缓存**：所有读写操作命中 `_cache`，不落盘，避免每次请求的 IO 开销
- **定期持久化**：`update_preferences` / `add_correction` 仅标记 `_dirty=True`，后台 5 分钟定时器自动刷新到 SQLite
- **会话结束强制落盘**：`on_session_end()` 取消定时器、立即 `_flush_to_db()`，保证数据不丢失
- SQLite WAL 模式 + NORMAL 同步，支持高并发多会话写入，无文件锁竞争
- 超过 30 天的记录自动清理，防止数据无限膨胀

#### 2. 修正历史（Few-shot）

- 记录用户对 AI 回答的修正 → 存为 few-shot 示例
- 最多保留 20 条，按时间排序
- 可生成修正提示注入 system prompt，帮助 AI 避免重复错误

#### 3. 会话生命周期

- `sync_from_short_term()`: 会话中实时同步实体到长期记忆缓存
- `on_session_end()`: 会话结束时更新统计（会话数、总轮次），取消定时器并立即落盘
- `close()`: 显式关闭，取消定时器 + 强制落盘，`__del__` 自动调用

### 长期记忆（LongTermMemory）

#### 1. 用户偏好统计（时间衰减加权 + 写缓存）

```python
from memory import ShortTermMemory

memory = ShortTermMemory(max_rounds=5, decay_factor=0.9, summarize_fn=my_llm_summarize)
```

| 方法                                                    | 说明                                                                          |
| ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `add_turn(sid, user_msg, assistant_msg, entities=None)` | 记录一轮对话，超 max_rounds 时自动触发 LLM 摘要                               |
| `add_entities(sid, **entities)`                         | 单独追加实体，`add_entities(sid, category="手机", brand="Apple")`             |
| `get_history(sid)`                                      | 获取对话历史（list[dict]）                                                    |
| `get_last_turn(sid)`                                    | 获取最近一轮对话                                                              |
| `get_entities(sid)`                                     | 获取合并后的实体（跨轮次累积，向后兼容）                                      |
| `get_summaries(sid)`                                    | 获取已淘汰轮次的 LLM 摘要列表                                                 |
| `resolve_context(sid, query)`                           | 槽位填充 + 置信度衰减：缺失槽位自动从历史填充，输出含权重（如 `Apple(0.81)`） |
| `build_messages(sid, system_prompt, user_msg)`          | 组装完整 messages 列表（含摘要注入到 system 消息）                            |
| `clear(sid)`                                            | 清除指定会话的记忆（含摘要）                                                  |
| `active_sessions()`                                     | 当前活跃会话数                                                                |
| `session_info(sid)`                                     | 获取会话详情（含 `summary_count`）                                            |
| `cleanup_expired(max_age_seconds=3600)`                 | 清理超时会话，默认1小时                                                       |

### LongTermMemory

```python
from memory import LongTermMemory

ltm = LongTermMemory(data_dir="data", user_id="default", half_life_days=7.0, max_age_days=30.0, flush_interval=300.0)
```

| 方法                                                  | 说明                                                           |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| `update_preferences(entities)`                        | 追加偏好记录（带时间戳），支持 `{key: val}` 或 `{key: [list]}` |
| `get_preferences()`                                   | 获取时间衰减加权后的偏好（top5 + 加权得分）                    |
| `get_preference_prompt()`                             | 生成偏好提示文本，用于注入 system prompt                       |
| `add_correction(question, original, corrected)`       | 记录一条修正历史（few-shot 示例）                              |
| `get_corrections(limit=5)`                            | 获取最近 N 条修正历史                                          |
| `get_correction_prompt()`                             | 生成修正参考提示，用于注入 system prompt                       |
| `sync_from_short_term(short_term_memory, session_id)` | 从短期记忆同步实体到长期记忆                                   |
| `on_session_end(short_term_memory, session_id)`       | 会话结束时更新统计并保存                                       |
| `get_stats()`                                         | 获取统计信息（会话数、总轮次）                                 |
| `clear()`                                             | 清空当前用户的长期记忆（缓存 + 数据库）                        |
| `clear_file()`                                        | 删除 SQLite 数据库文件                                         |
| `close()`                                             | 取消定时器 + 强制落盘，优雅关闭                                |

**数据库** (`data/ltm.db`):

```sql
-- 偏好记录：每条独立存储，get_preferences() 实时计算指数衰减加权得分
preferences(user_id, key, value, timestamp)

-- 修正历史：few-shot 示例，保留最近 20 条
corrections(user_id, id, question, original, corrected, timestamp)

-- 统计：每次会话结束更新
stats(user_id, total_sessions, total_turns, last_updated)
```

> **持久化策略**：所有写入先更新内存 `_cache`，仅标记 `_dirty=True`。后台 `threading.Timer` 每 `flush_interval` 秒（默认 300s）检查一次，若脏则 `DELETE + INSERT` 全量刷新到 SQLite。`on_session_end()` 取消定时器立即落盘。旧 JSON 格式（`data/{user_id}.json`）首次加载时自动迁移至 SQLite 并重命名为 `.bak`。

## 集成示例

### 1. 初始化

```python
from memory import ShortTermMemory, LongTermMemory

stm = ShortTermMemory(max_rounds=5, summarize_fn=my_summarize_fn)
ltm = LongTermMemory(user_id="user_001")
```

### 2. 获取会话标识

```python
session_id = "session_xxx"  # 由调用方传入，如 WebSocket 连接 ID、HTTP Session ID 等
```

### 3. 上下文消解

```python
enhanced_query = stm.resolve_context(session_id, message)
# 用 enhanced_query 替代原始 message 进行后续处理
```

### 4. 记录对话（附带实体）

```python
entities = {"category": "电子产品", "brand": "Apple", "topic": "电池续航"}
stm.add_turn(session_id, message, reply_text, entities=entities)
ltm.sync_from_short_term(stm, session_id)
```

### 5. 构建多轮 messages

```python
messages = stm.build_messages(session_id, system_prompt, user_message)
```

### 6. 偏好注入

```python
preference_prompt = ltm.get_preference_prompt()
correction_prompt = ltm.get_correction_prompt()
full_system_prompt = system_prompt + "\n\n" + preference_prompt + "\n\n" + correction_prompt
```

### 7. 清空会话

```python
stm.clear(session_id)
# 长期记忆保留，跨会话偏好不丢失
```

## 测试

### 基础多轮对话测试

使用 `chat_test.py` 在终端中测试多轮对话记忆功能：

```bash
cd app/memory
python chat_test.py                # 普通模式
python chat_test.py --debug        # 调试模式（显示上下文消解和记忆状态）
```

### 内置命令

| 命令            | 说明                                                |
| --------------- | --------------------------------------------------- |
| `exit` / `quit` | 退出（自动触发 `on_session_end`，更新长期记忆统计） |
| `clear`         | 清空当前会话短期记忆                                |
| `info`          | 查看短期记忆状态 + 长期记忆统计（轮次、实体、历史） |
| `ltminfo`       | 查看长期记忆详情（偏好、统计、偏好提示文本）        |

### 启动时注入

- 长期记忆的偏好提示自动注入 system prompt（如"用户常用 brand: Apple, category: 手机"）
- 退出时自动更新 `data/ltm.db`：会话数 +1，轮次数累加

### 测试场景

1. **槽位填充**：先问"推荐一款Apple手机"（附带实体），再问"那它的电池怎么样" → 第二问检测"它"指代，自动填充缺失槽位，输出含权重
2. **追问补全**：先问"推荐一款Apple手机"，再问"有黑色吗" → 短查询触发追问检测，自动补全品类和品牌，显示衰减权重
3. **置信度衰减**：第1轮设实体，第3轮再追问 → 权重从 1.00 衰减到 0.81，可视化确认衰减有效
4. **槽位去重**：查询中已有"Apple"则不重复填充"brand:Apple"
5. **智能摘要**：连续6轮以上，检查淘汰的旧轮次是否生成了 LLM 摘要，`info` 命令可查看
6. **长期记忆持久化**：退出后重启，检查 `ltminfo` 是否保留了之前的会话统计

### 集成长期记忆

```python
from memory import ShortTermMemory, LongTermMemory

stm = ShortTermMemory(max_rounds=5, summarize_fn=summarize_fn)
ltm = LongTermMemory(user_id="user_001", half_life_days=7.0, max_age_days=30.0, flush_interval=300.0)

def respond(message, history, session_id):
    # 上下文消解
    enhanced_query = stm.resolve_context(session_id, message)

    # ... 业务逻辑 ...

    # 记录对话和实体
    entities = {"category": "电子产品", "brand": "Apple", "topic": "电池续航"}
    stm.add_turn(session_id, message, reply_text, entities=entities)
    ltm.sync_from_short_term(stm, session_id)

    return reply_text

def clear_session(session_id):
    stm.clear(session_id)
    ltm.close()  # 确保落盘
    # 长期记忆保留，跨会话偏好不丢失
    return []
```

## 后续迭代方向

1. **自动学习修正**：从用户交互中自动识别修正，无需手动调用 `add_correction`
2. **记忆检索**：将历史对话向量化，按语义相似度召回相关历史
