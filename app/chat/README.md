# Chat 模块 — ReAct 多轮推理对话

基于 `memory` 模块的短期/长期记忆，实现 **ReAct 循环推理对话**，支持流式输出。

提供两种模式：

- **纯推理模式**：不依赖检索，仅靠 LLM 多步推理
- **带检索模式**：集成 RAG 多路召回 + Rerank，可调用报告数据库检索

---

## 模块结构

```
chat/
├── chat.py              # ReAct 纯推理对话终端（可独立运行）
├── rag_chat.py          # ReAct + RAG 检索对话终端（可独立运行）
└── README.md            # 本文件
```

---

## 设计决策

| 决策项   | 选择                                      | 说明                                                                   |
| -------- | ----------------------------------------- | ---------------------------------------------------------------------- |
| 推理模式 | ReAct 循环                                | LLM 每轮输出推理/动作 → 系统执行（检索） → LLM 看到结果再判断          |
| 输出格式 | `[CONTINUE]`/`[ACTION: search]`/`[FINAL]` | 简单标记，LLM 容易遵守，`parse_react_output()` 统一解析                |
| 输出方式 | 流式 SSE                                  | 边生成边显示，用户实时看到 ReAct 推理过程                              |
| 消息角色 | 增量构建 + 角色交替                       | 推理 → assistant，检索结果 → user（模拟"环境反馈"），确保对话结构正确  |
| 终止条件 | LLM 自主判断                              | 每轮展示完整对话历史，让 LLM 自己决定是否充分                          |
| 最大步数 | 5 步                                      | 防止无限循环，超限后基于已有对话历史追加提示，强制输出最终回答         |
| 记忆集成 | 依赖 memory 模块                          | 继承 STM 的智能摘要、上下文消解、实体追踪；LTM 的时间衰减偏好统计      |
| RAG 检索 | 复用现有管线                              | 复用 `retrieval.py` + `rerank.py`，保持架构一致                        |
| 网络容错 | `@retry()` 装饰器                         | `get_embedding`、`chat_stream`、`rerank`、`summarize_fn` 均带 3 次重试 |
| 资源管理 | `MilvusClient` 单例                       | `main()` 中初始化一次，所有检索复用，退出时 `close()`                  |

---

## 核心功能

### 1. 纯推理模式（chat.py）

```
用户输入: "脑出血和高血压有什么关系"
    │
    ▼
[第1步] LLM 流式输出 → [CONTINUE] 推理内容1
    │
    ▼
[第2步] LLM 看到完整对话历史 → 判断 → [CONTINUE] 推理内容2
    │
    ▼
[第3步] LLM 综合判断 → [FINAL] 最终回答
    │
    ▼
写入 STM + LTM
```

### 2. 带检索模式（rag_chat.py）

```
用户输入: "左基底节区高密度影和高血压有关系吗"
    │
    ▼
[第1步] LLM 流式输出 → [CONTINUE] 推理，分析问题
    💭 推理: 左基底节区高密度影常见于...
    messages ← assistant(推理文本)
    │
    ▼
[第2步] LLM 判断需要检索 → [ACTION: search] 左基底节区 高密度影 高血压
    🔍 检索: 左基底节区 高密度影 高血压
    messages ← assistant(模型输出)
    🔍 执行检索：向量多路召回 + Rerank → 返回参考报告
    messages ← user("观察（检索结果）：...\n请判断下一步。")
    │
    ▼
[第3步] LLM 看到检索结果 → [CONTINUE] 分析
    💭 推理: 根据检索结果，参考1显示...
    messages ← assistant(推理文本)
    │
    ▼
[第4步] LLM 综合推理 → [FINAL] 最终回答
    ✅ 回答: 根据参考报告分析，左基底节区...
    │
    ▼
写入 STM + LTM
```

### 3. 消息角色管理（关键设计）

ReAct 循环中每轮 LLM 输出后，消息按以下规则增量追加到对话历史：

| LLM 输出                  | 追加角色    | 追加内容                                       |
| ------------------------- | ----------- | ---------------------------------------------- |
| `[CONTINUE]` 推理文本     | `assistant` | 推理文本（不含 `[CONTINUE]` 标签）             |
| `[ACTION: search]` 检索词 | `assistant` | 完整模型输出                                   |
| ↓ 检索完成后              | `user`      | `"观察（检索结果）：{result}\n请判断下一步。"` |
| `[FINAL]` 最终回答        | `assistant` | 完整模型输出，循环终止                         |

设计要点：

- **推理**作为 `assistant` 角色，是 LLM 的"思考"
- **检索结果**作为 `user` 角色注入，模拟"环境反馈"，确保角色交替不出现连续 assistant
- 消息**增量追加**而非每轮重建，避免 token 膨胀和上下文丢失

### 4. 流式输出

- 使用 SSE（Server-Sent Events）协议，`stream: True`
- `iter_lines` 逐块解析，实时打印模型输出，用户可看到完整的 ReAct 推理过程
- 解析后根据 `parse_react_output()` 返回类型打印对应状态标签：
  - `💭 推理: {前50字}...` — 推理步骤
  - `🔍 检索: {检索词}` — 执行检索
  - `✅ 回答: {完整回答}` — 最终回答

### 5. 网络重试机制

所有网络调用均通过 `@retry(max_attempts=3, delay=2)` 装饰器保护：

| 函数                  | 调用目标           | 说明               |
| --------------------- | ------------------ | ------------------ |
| `get_embedding()`     | Embedding API      | 文本向量化         |
| `chat_stream()`       | Chat API（流式）   | LLM 推理           |
| `rerank_with_retry()` | Rerank API         | Cross-Encoder 精排 |
| `summarize_fn()`      | Chat API（非流式） | 短期记忆摘要生成   |

临时网络失败自动重试，3 次均失败才抛出异常，避免单次波动导致崩溃。

### 6. 记忆集成

- **短期记忆**：上下文消解（补充缺省实体）、智能摘要（淘汰轮次压缩）
- **长期记忆**：偏好统计（时间衰减加权）、会话生命周期管理
- **异常安全**：`try...finally` 包裹主循环，确保即使推理异常也能保存已生成的回答到 STM + LTM

---

## 工作流对比

| 阶段 | 纯推理模式 `chat.py`                  | 带检索模式 `rag_chat.py`                                     |
| ---- | ------------------------------------- | ------------------------------------------------------------ |
| 初始 | `[CONTINUE]` 推理                     | `[CONTINUE]` 推理 或 `[ACTION: search]` 检索（LLM 自主判断） |
| 推理 | 每步附加 assistant 消息，保持对话历史 | 每步附加 assistant 消息，保持对话历史                        |
| 工具 | 无                                    | RAG 检索自动执行，结果以 user 消息注入（"观察"）             |
| 终止 | LLM 输出 `[FINAL]`                    | LLM 输出 `[FINAL]`                                           |
| 超限 | 达 5 步后追加提示，强制输出 `[FINAL]` | 达 5 步后基于完整对话历史追加提示，强制输出 `[FINAL]`        |
| 容错 | `@retry()` 保护 LLM 调用              | `@retry()` 保护 LLM 调用 + Embedding + Rerank + 摘要         |

---

## 内置命令

两种模式都支持以下命令：

| 命令        | 说明                       |
| ----------- | -------------------------- |
| `exit/quit` | 退出程序，自动保存长期记忆 |
| `clear`     | 清空当前会话的短期记忆     |
| `info`      | 查看短期/长期记忆状态      |
| 直接输入    | 进入 ReAct 推理循环        |

---

## 使用方式

### 运行纯推理模式

```bash
cd app/chat
python chat.py           # 普通模式，每步显示推理过程
python chat.py --debug   # 调试模式，显示 LLM 输入输出细节
```

### 运行带检索模式

```bash
cd app/chat
python rag_chat.py           # 普通模式，流式显示 ReAct 推理 + 检索过程
python rag_chat.py --debug   # 调试模式，额外显示完整 messages 和检索结果
```

要求：

- `milvus_lite.db` 必须存在（在 `app/` 目录下）
- `.env` 中配置 `EMBED_URL`, `CHAT_URL`, `CHAT_MODEL` 等环境变量

---

## 依赖

```
chat/ ──依赖──▶ memory/
chat/ ──依赖──▶ retrieval.py (多路召回)
chat/ ──依赖──▶ rerank.py (精排)
chat/ ──依赖──▶ query_rewrite.py (关键词解析)
memory/ 完全独立，不依赖 chat/
```

---

## 关键函数

### `parse_react_output(text: str) -> tuple[str, str | tuple]`

统一解析 LLM 的 ReAct 输出，返回 `(类型, 内容)`：

- `[FINAL]` → `("final", "回答文本")`
- `[ACTION: search]` → `("action", ("search", "检索词"))`
- `[CONTINUE]` → `("continue", "推理文本")`
- 无标签 → `("continue", "原始文本")`（默认当作推理）

使用 `re.IGNORECASE | re.DOTALL` 匹配，支持大小写不敏感和多行输出。

### `retry(max_attempts=3, delay=2, exceptions=(RequestException,))`

装饰器工厂，为网络调用函数添加自动重试。失败时打印 `⚠️` 警告并等待后重试。

### `chat_stream(messages, max_tokens=2048, temperature=0.3, debug=False)`

流式调用 LLM。`debug=True` 时边生成边打印，`debug=False` 时静默收集。返回完整文本。

---

## 后续迭代

- [ ] 推理步骤可视化（Web UI）
- [ ] 中断恢复（用户可中途打断 LLM 推理）
- [ ] 并行推理（多个推理分支同时探索）
- [ ] 支持更多工具类型（除了 search 还可扩展计算、改写等）
