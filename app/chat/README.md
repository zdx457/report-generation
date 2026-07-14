# Chat 模块 — Tool Calling 对话

基于 `memory` 模块的短期/长期记忆，实现 **Tool Calling 架构对话**，支持流式输出。

提供三种模式：

- **Tool Calling 模式**：`rag_chat_v2.py` — LLM 自主决策工具调用（检索/编辑/重写/闲聊），支持 FastAPI Web 服务
- **ReAct 推理模式**：`chat.py` — 纯推理，不依赖检索（保留兼容）
- **ReAct + RAG 模式**：`rag_chat.py` — ReAct + RAG 检索（保留兼容）

---

## 模块结构

```
chat/
├── rag_chat_v2.py       # ★ Tool Calling 架构主入口 + FastAPI Web 服务
├── chat.py              # ReAct 纯推理对话终端（保留兼容）
├── rag_chat.py          # ReAct + RAG 检索对话终端（保留兼容）
└── README.md            # 本文件
```

---

## 设计决策

| 决策项   | 选择                        | 说明                                                                                     |
| -------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| 决策模式 | Tool Calling（OpenAI 兼容） | LLM 自主决定调用哪个工具（rag_search / edit_report / refine_report），替代硬编码意图分类 |
| 工具注册 | ToolRegistry                | 统一注册、Schema 管理、执行，支持动态扩展                                                |
| 输出方式 | 流式 SSE                    | 二次 LLM 调用结果流式推送，用户实时看到生成过程                                          |
| 消息角色 | assistant + tool            | 第一轮返回 tool_calls（assistant），工具结果追加 role:tool，第二轮生成最终回复           |
| 记忆注入 | 工具执行前注入              | rag_search / refine_report 工具内部自动注入 LTM 偏好 + Entity 上下文                     |
| 最大轮数 | 6 轮对话历史                | 历史消息压缩后传入 LLM 上下文                                                            |
| 网络容错 | `@retry()` 装饰器           | `get_embedding`、`chat_stream`、`chat_with_tools`、`rerank` 均带 3 次重试                |
| 资源管理 | `MilvusClient` 单例         | `_get_or_create_session()` 中初始化一次，所有检索复用                                    |

---

## 核心功能

### 1. Tool Calling 模式（rag_chat_v2.py）

```
用户输入: "CT脑出血"
    │
    ▼
[Phase 1] 预处理
    ├ 实体提取（LLM + 规则双引擎）
    ├ 意图检测（new_session / append / switch）
    ├ 上下文消解（补全省略信息）
    └ ★ 如果 selected_diagnosis 非空（用户点击歧义按钮）→ 跳过 Phase 1，保留 modality/body_part
    │
    ▼
[Phase 2] Tool Calling 主循环
    │
    ├── 构建 System Prompt
    │    注入：LTM 偏好 + Entity 上下文 + 上轮报告
    │
    ├── chat_with_tools(messages, tools=[rag_search, edit_report, refine_report])
    │     → LLM 返回 tool_calls: [{"name": "rag_search", "arguments": {...}}]
    │
    ├── 执行 rag_search 工具
    │     ├ 多路召回（向量 + 元数据 + 关键词）
    │     ├ Rerank 精排
    │     ├ 歧义检测：top-N 分数接近且诊断多样 → 追问用户
    │     │   ├ 缓存检索结果 + Rerank 结果（全局 dict，跨请求存活）
    │     │   └ _emit("ambiguous") → 前端按钮选择
    │     ├ ★ 如果 selected_diagnosis 非空（用户点击按钮）：
    │     │   ├ 命中缓存 → 跳过检索和 Rerank
    │     │   ├ _filter_by_selected_diagnosis() → 过滤只保留匹配的诊断
    │     │   └ _rebuild_search_result() → 重建检索文本
    │     ├ 注入 LTM 偏好 + Entity 上下文
    │     └ LLM 生成结构化报告 JSON
    │
    ├── 追加 role:tool 消息到 messages
    │
    └── 报告类结果（_is_final=true）→ 跳过二次 LLM，直接发送报告
```

### 2. 三个工具

| 工具            | 文件                   | 功能                      | 特点                            |
| --------------- | ---------------------- | ------------------------- | ------------------------------- |
| `rag_search`    | `tools/rag_tool.py`    | RAG 检索 + 结构化报告生成 | 注入 LTM + Entity，合并新旧报告 |
| `edit_report`   | `tools/edit_tool.py`   | 修改已有报告              | 保留 Key 结构，按指令修改       |
| `refine_report` | `tools/refine_tool.py` | 重写报告风格              | 跳过 RAG 检索，保留医学内容     |

### 3. Tool Calling 调用流程

```
第一次 LLM 调用（带 tools）:
  POST /v1/chat/completions
  {
    "messages": [...],
    "tools": [{"type": "function", "function": {...}}, ...],
    "stream": false
  }
  → 返回: { "choices": [{"message": {"tool_calls": [...]}}] }

执行工具:
  registry.execute(tool_call_id, name, arguments)
  → 返回工具执行结果字符串

第二次 LLM 调用（不带 tools）:
  POST /v1/chat/completions
  {
    "messages": [
      ...原有消息,
      {"role": "assistant", "tool_calls": [...]},
      {"role": "tool", "tool_call_id": "...", "content": "工具结果"},
    ],
    "stream": true
  }
  → 流式返回最终回复
```

### 4. 错误处理

- **工具执行失败**: 返回 `{"error": "..."}` JSON 给 LLM，LLM 可据此重试或向用户解释
- **缺少 current_report**: 对 edit_report / refine_report，自动从 last_report 注入
- **JSON 解析失败**: 兜底返回 raw 文本，不中断流程
- **Key 集合保护**: 编辑/重写后如果 Key 集合变化，自动回退旧报告

### 5. 记忆集成

- **短期记忆**：上下文消解（补充缺省实体）、智能摘要（淘汰轮次压缩）
- **长期记忆**：偏好统计（时间衰减加权）、会话生命周期管理
- **Entity Tracker**：模态/部位槽位、意图检测、上下文消解

---

## 工作流对比

| 阶段     | 旧版（三段式）`rag_chat_v2.py` 旧                | 新版（Tool Calling）`rag_chat_v2.py` |
| -------- | ------------------------------------------------ | ------------------------------------ |
| 决策     | `classify_intent()` 硬编码 5 分类                | `chat_with_tools()` LLM 自主决策     |
| 路由     | `if/elif` 分叉 (SEARCH/EDIT/REFINE/CHAT/CONFIRM) | Tool Calling 自动路由                |
| 扩展     | 新增意图需修改 prompt + 代码                     | 新增工具只需注册到 ToolRegistry      |
| 工具结果 | 代码直接处理                                     | 以 role:tool 消息返回 LLM，二次推理  |
| 闲聊     | 独立的 chat_reply 分支                           | LLM 无 tool_calls 时直接回复         |

---

## 内置命令

| 命令        | 说明                       |
| ----------- | -------------------------- |
| `exit/quit` | 退出程序，自动保存长期记忆 |
| `clear`     | 清空当前会话的短期记忆     |
| 直接输入    | 进入 Tool Calling 推理循环 |

---

## 使用方式

### 运行 Web 服务

```bash
cd app/chat
python rag_chat_v2.py --web
```

浏览器访问 `http://localhost:8000`

### 运行 CLI 模式

```bash
cd app/chat
python rag_chat_v2.py
```

要求：

- `milvus_lite.db` 必须存在（在 `app/data_pipeline/` 目录下）
- `.env` 中配置 `CHAT_URL`, `CHAT_MODEL` 等环境变量

---

## 依赖

```
chat/ ──依赖──▶ tools/          (Tool Calling 工具注册与执行)
chat/ ──依赖──▶ memory/         (STM + LTM + EntityTracker)
chat/ ──依赖──▶ rag/retrieval.py (多路召回)
chat/ ──依赖──▶ rag/rerank.py   (精排)
chat/ ──依赖──▶ rag/query_rewrite.py (关键词解析)
tools/ ──依赖──▶ rag/retrieval.py
tools/ ──依赖──▶ prompt/        (structure.md, edit.md, refine.md)
memory/ 完全独立，不依赖 chat/
```

---

## 关键函数

### `chat_with_tools(messages, tools=None, max_tokens=512, temperature=0.3, debug=False)`

非流式 LLM 调用，支持 Tool Calling。返回 `(content_text, tool_calls_list)` 元组：

- `content_text`: 文本回复（可能为 None）
- `tool_calls_list`: `[{"id": str, "name": str, "arguments": dict}]` 或 None

### `chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=False, caller="chat_stream")`

流式调用 LLM，被工具内部使用。`_emit` 不为 None 时逐 token 推送，为 None 时静默收集返回完整文本。

### `_build_tool_registry(ltm, entity_tracker, client, last_report, _emit)`

构建 ToolRegistry，注册三个工具：

- `rag_search` — 注入 chat_fn, ltm, entity_tracker, get_embedding_fn, search_reports_fn
- `edit_report` — 注入 chat_fn
- `refine_report` — 注入 chat_fn, ltm, entity_tracker

### `run_pipeline(query, session_id, stm, entity_tracker, ltm, client, last_report, _emit)`

Tool Calling 架构主流程：

1. Phase 1: 预处理（实体提取、意图检测、上下文消解、查询改写）
2. Phase 2: Tool Calling 主循环（构建 registry → 第一次 LLM 调用 → 执行工具 → 第二次 LLM 调用）
3. Phase 3: 后处理（更新 last_report、STM 记录）

---

## 后续迭代

- [ ] 工具并行调用（多个 tool_calls 并发执行）
- [ ] 用户中断工具执行
- [ ] 工具调用过程可视化（Web UI 展示调用链）
- [ ] 支持更多工具类型（如 calculate、compare、translate 等）
- [ ] 工具调用结果缓存
