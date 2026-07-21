# 医疗影像报告生成Agent v2

基于 RAG（检索增强生成）的医学影像报告结构化输出系统，通过 **Tool Calling 架构**让 LLM 自主决定调用工具（检索生成 / 编辑修改 / 风格重写 / 直接回复），将用户输入的检查信息转化为规范的影像学报告。

## 核心特性

- **Tool Calling 架构**：LLM 自主决策调用哪个工具（`rag_search` / `edit_report` / `refine_report`），替代硬编码意图分类，架构更灵活可扩展
- **PromptBuilder 统一拼装**：消除 `rag_tool.py`、`refine_tool.py`、`rag_chat_v2.py` 中重复的 Prompt 拼装代码，LTM 偏好 + Entity 上下文 + 模板 + Last Report 由 `PromptBuilder.build()` 统一管理，拼接顺序和 reasoning 去除逻辑单点维护
- **三层记忆架构**：Entity Tracker（实体槽位）+ STM（短期记忆）+ LTM（长期偏好），独立解耦
- **LLM 实体提取**：双引擎机制（LLM JSON 结构化提取优先级 + 规则兜底），关键词按长度降序匹配，支持多部位（如 `CT 肝脏和胆囊` → `body_part: ["肝脏", "胆囊"]`）
- **意图切换清洗**：切换检查部位时强制清空 STM 和 last_report，杜绝旧病灶残留
- **多轮对话**：上下文消解 + 指代消解，继承检查类型，多轮报告自动合并去重
- **短期记忆**：可配置轮数（前端配置）+ 对话摘要，可通过前端面板查看
- **Agent 思考过程**：前端可视化展示工具调用、查询改写、召回、Rerank、推理全流程
- **流式输出**：FastAPI + SSE 推送，报告和闲聊消息实时流式渲染
- **配置统一管理**：`config.yml` 一站式配置，环境变量自动覆盖
- **Rerank 降级**：Rerank 服务不可用时自动降级为纯向量检索
- **知识库管理**：前端上传 xlsx → 自动切片 → 提取元数据 → 构建向量库，全流程可视化
- **配置管理**：前端可视化编辑 `config.yml`，在线测试模型连通性，保存即生效
- **会话持久化**：基于 SQLite 的 SessionStore，对话历史/实体槽位/上一轮报告自动保存，断线恢复、历史会话管理
- **记忆检索注入**：基于语义相似度的按需检索，替代全量注入 LTM/STM，减少无关记忆干扰，降低 Token 消耗
- **歧义检测与用户选择**：同病不同修饰（如"脑出血"vs"基底节区脑出血"vs"脑出血（破入脑室）"）时，前端弹出按钮让用户精确选择，避免 LLM 猜测
- **歧义缓存加速**：用户点击歧义选项后，复用上次的检索+Rerank 结果，跳过昂贵的多路召回和重排序，显著降低响应延迟
- **部位补全增强**：当 `parse_query_keywords()` 从文本中提取不到部位时（如"CT 脑出血"中"脑"不是独立部位词），自动从 Entity Tracker 的实体槽位和诊断关键词启发式推断中两层补全部位，提升元数据过滤和关键词检索精度，不修改向量检索 query 文本以保证语义相似度不受影响

## 系统架构

```
用户输入 "CT脑出血"
       │
       ▼
┌──────────────────────────┐
│  预处理                   │
│  ├ 模糊检测 → 追问       │
│  ├ 缺少模态拦截 → 追问    │
│  ├ 上下文消解 → 指代继承  │
│  ├ 术语标准化             │
│  └ 查询改写               │
└──────────┬───────────────┘
           ▼
┌──────────────────────────────────────────────────┐
│  Tool Calling 主循环                              │
│                                                   │
│  1. MemoryRetriever: 检索最相关 LTM/STM 片段 → 构建 System Prompt │
│  2. chat_with_tools(messages, tools=[...])        │
│     → LLM 自主决策：调用工具 or 直接回复          │
│                                                    │
│     ┌──────────────┬──────────────┬──────────────┐ │
│     ▼              ▼              ▼              ▼ │
│  rag_search    edit_report   refine_report   直接回复│
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────┐ │
│  │多路召回 │    │修改报告│    │重写风格│    │闲聊│ │
│  │Rerank  │    │保留Key │    │跳过检索│    │回复│ │
│  │结构化生成│   │按指令改│    │保留内容│    └────┘ │
│  └────────┘    └────────┘    └────────┘            │
│                                                    │
│  3. 工具返回 _is_final: true → 跳过二次 LLM 调用    │
│     直接发送报告到前端，避免引入幻觉                │
│     非 final 结果（错误/文本）→ 二次 LLM 包装回复    │
└──────────┬───────────────────────────────────────┘
           ▼
┌──────────────────────────────────────┐
│  存储 / 记忆更新                       │
│  ├ STM: add_turn                    │
│  ├ EntityTracker: 槽位更新           │
│  ├ LTM: 偏好写入                     │
│  ├ last_report 更新                 │
│  ├ ★ MemoryRetriever: 语义检索注入  │
│  └ ★ SessionStore: SQLite 持久化    │
│     对话记录 + 实体槽位 + 报告       │
└──────────────────────────────────────┘
```

## 项目结构

```
app/
├── chat/
│   ├── rag_chat_v2.py     # ★ 主入口：Tool Calling 架构 + FastAPI Web 服务
│   ├── rag_chat.py        # 旧版（保留兼容）
│   ├── chat.py            # 命令行版
│   └── README.md          # chat 模块说明
├── tools/                 # ★ 工具模块（Tool Calling）
│   ├── registry.py        # 工具注册中心（ToolResult 数据类 + OpenAI 兼容 schema）
│   ├── utils.py           # 工具共用辅助函数（JSON提取）
│   ├── rag_tool.py        # rag_search — RAG 检索 + 结构化报告生成
│   ├── edit_tool.py       # edit_report — 修改已有报告
│   ├── refine_tool.py     # refine_report — 重写报告风格
│   └── __init__.py
├── rag/
│   ├── retrieval.py       # 多路召回模块（向量检索 + 元数据过滤 + 关键词检索）
│   ├── query_rewrite.py   # 查询改写模块（模糊检测 + 术语标准化 + 关键词解析 + LLM 改写）
│   ├── rerank.py          # Rerank 精排模块（SiliconFlow API）
│   └── prompt.md          # 影像报告结构化规范（实际 prompt 模板在 app/prompt/ 目录）
├── memory/
│   ├── short_term.py      # 短期记忆（对话历史 + 自动摘要）
│   ├── long_term.py       # 长期记忆（持久化偏好存储）
│   ├── entity_tracker.py  # 独立实体追踪（模态/部位槽位 + 意图判断 + 上下文消解）
│   ├── chat_test.py       # 记忆模块测试脚本
│   ├── data/              # 长期记忆数据库文件（ltm.db）
│   └── README.md          # 记忆模块说明
├── data_pipeline/
│   ├── build_vector_db.py # 向量数据库构建（支持增量/全量重建）
│   ├── extract_metadata.py# 从 xlsx 提取标准术语（生成 metadata.json）
│   ├── xlsx_slicer.py     # xlsx 按行切片工具
│   ├── milvus_lite.db/    # Milvus Lite 向量数据库（运行时生成）
│   ├── xlsx_slices/       # 切片后的 Markdown 文件
│   └── report_template/   # 原始 xlsx 报告模板 + metadata.json
├── config.py              # 配置读取器（config.yml → 各模块）
├── prompt/                # 系统提示词模板（各阶段 LLM prompt）
│   ├── builder.py         # ★ PromptBuilder — 统一 Prompt 拼装器（LTM + Entity + 模板 + Last Report）
│   ├── __init__.py        # load_prompt() 模板加载器
│   ├── tool_orchestrator.md # ★ Tool Calling 编排提示词（指导 LLM 决策工具调用）
│   ├── structure.md       # 报告结构化提取（检索结果 → JSON）
│   ├── edit.md            # 编辑已有报告
│   ├── refine.md          # 重写报告风格
│   ├── entity_extraction.md # 实体提取（LLM JSON 结构化提取 prompt）
│   ├── chat.md            # 闲聊回复
│   ├── intent.md          # 意图识别（旧版，保留兼容）
│   ├── report_generation.md # 报告生成（综合 prompt）
│   ├── rewrite.md         # 查询改写
│   ├── summarize.md       # 对话摘要
│   ├── react_system.md    # ReAct 系统 prompt
│   ├── react_simple.md    # ReAct 简化 prompt
│   └── multi_disease.md   # 多病种报告
├── store/                 # ★ 会话持久化（Phase 4）
│   ├── session_store.py   # SessionStore — SQLite 持久化存储
│   └── __init__.py
├── test/                  # 测试与评估脚本
│   ├── test_memory.py              # 记忆模块单元测试（意图切换/实体提取/LTM注入/上下文消解）
│   ├── test_phase2_verification.py # Phase 2 验证测试（LLM提取/脏数据/异常回退/向后兼容）
│   ├── _test_search.py             # 向量检索快速测试
│   ├── eval_retrieval.py           # 检索命中率评估
│   ├── eval_rerank_compare.py      # 向量检索 vs Rerank 对比评估
│   ├── test_rerank_flow.py         # 完整流程测试
│   └── tests_api/                  # API 连通性测试
└── README.md              # 本文件
```

## 配置说明

所有配置统一在项目根目录的 `config.yml` 中管理，通过 `config.py` 读取。优先级：**环境变量 > config.yml > 默认值**。敏感信息（如 API Key）建议通过环境变量传入。

## 使用方式

### Web 界面

```bash
python python_start.py
```

浏览器访问 `http://localhost:8000`，前端提供：

| 功能           | 说明                                                                 |
| -------------- | -------------------------------------------------------------------- |
| 聊天输入       | 输入检查信息，如 `CT 脑出血`、`MR 膝关节`                            |
| 结构化报告     | 影像学表现 + 诊断意见，实时流式渲染                                  |
| 历史对话       | 左侧栏展示所有对话记录，支持切换、删除，对话自动保存到 SQLite 持久化 |
| 短期记忆       | 弹窗查看当前会话的对话历史、实体、摘要                               |
| Agent 思考过程 | 折叠面板，展示工具调用、检索查询、召回、Rerank、推理                 |
| 清空会话       | 重置会话状态和短期记忆                                               |
| 知识库管理     | 上传 xlsx 报告模板、在线切片、提取元数据、构建向量库                 |
| 配置管理       | 可视化编辑模型配置（LLM/Embedding/Rerank），在线测试连通性           |

### 命令行

```bash
python app/chat/rag_chat_v2.py
```

## 知识库管理

前端提供完整的知识库（Excel 报告模板）管理界面，支持以下操作：

| 操作       | 说明                                                                                   | 对应 API                        |
| ---------- | -------------------------------------------------------------------------------------- | ------------------------------- |
| 上传模板   | 上传 `.xlsx` 格式的影像报告模板文件                                                    | `POST /api/kb/upload`           |
| 自动切片   | 上传后自动将 xlsx 每一行按列头切分为独立 Markdown 文件                                 | `xlsx_slicer.py`                |
| 提取元数据 | 从 xlsx 模板中提取标准术语（检查类型、部位、检查项目、诊断结论），生成 `metadata.json` | `POST /api/kb/extract-metadata` |
| 构建向量库 | 将切片后的 Markdown 文件向量化并入库，支持增量模式和全量重建                           | `POST /api/kb/build`            |
| 查看状态   | 查看当前向量库条目数、切片文件数、元数据状态                                           | `GET /api/kb/status`            |
| 查看文件   | 列出所有已上传的报告模板及其切片数量                                                   | `GET /api/kb/files`             |

```
上传 xlsx
   │
   ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ 自动切片     │ →  │ 提取元数据    │ →  │ 构建向量库    │
│ 按行转 .md  │    │ metadata.json │    │ bge-m3 向量化 │
└─────────────┘    └──────────────┘    └──────────────┘
```

## 配置管理

前端提供可视化配置面板，可在线编辑 `config.yml` 而不需要直接修改文件：

| 功能     | 说明                                                                         |
| -------- | ---------------------------------------------------------------------------- |
| 模型列表 | 管理多个 LLM / Embedding / Rerank 模型配置（base_url、model、api_key、参数） |
| 激活切换 | 从模型列表中选择当前使用的模型（通过 `active_models` 指定）                  |
| 连接测试 | 在线测试 LLM / Embedding / Rerank 的 API 连通性，显示响应预览                |
| 实时生效 | 保存配置后自动调用 `reload_config()`，无需重启服务                           |
| 安全保护 | API Key 在前端显示为掩码占位符，实际值从 `.env` 环境变量读取                 |

配置优先级：**环境变量 > config.yml > 默认值**

## 报告生成流程

完整流程分 3 个阶段：

```
用户输入 "脑梗"（第2轮，上轮是"CT 脑出血"）
       │
       ▼
┌─ Phase 1: 预处理 ────────────────────────────┐
│ 1. 模糊检测 → 不过于模糊，跳过              │
│ 2. 缺少模态拦截 → 有模态，跳过              │
│ 3. 上下文消解 → "脑梗" 缺少检查类型 → 继承  │
│    上轮的 "CT" → "CT 脑梗"                  │
│ 4. 术语标准化 → "CT 脑梗"                   │
│ 5. 查询改写 → 有诊断词，无需改写             │
│ 6. 获取历史 → 最近 6 轮对话                 │
└──────────────┬──────────────────────────────┘
               ▼
┌─ Phase 2: Tool Calling 主循环 ─────────────────┐
│                                                │
│  1. 构建 System Prompt                         │
│     PromptBuilder.build("tool_orchestrator",    │
│       LTM 偏好 + Entity 上下文)                  │
│     注入上一轮报告（供工具决策参考）              │
│                                                │
│  2. chat_with_tools(messages, tools=[          │
│       rag_search, edit_report, refine_report   │
│     ])                                         │
│     → LLM 自主决策：调用 rag_search            │
│                                                │
│  3. 执行 rag_search 工具                        │
│     ├ PromptBuilder.build("structure", ...)      │
│     │   统一拼装 LTM + Entity + 模板 + Last Report│
│     ├ 参数容错：缺失 modality/body_part 从      │
│     │   EntityTracker 自动补全                  │
│     ├ ★ 部位补全：parse_query_keywords() 提取不到部位时，│
│     │   从 entity_tracker 槽位 + 诊断关键词启发式推断│
│     │   （如 "CT 脑出血" → 部位补全为 "头颅"）   │
│     │   仅增强元数据/关键词路径，不修改向量检索文本│
│     ├ 流式状态：_emit("status") 推送            │
│     │   searching → generating → done           │
│     ├ 多路召回（向量 + 元数据 + 关键词）       │
│     ├ Rerank 精排 → Top-10                     │
│     ├ 歧义检测：top-10 分数接近且诊断多样时     │
│     │   → 缓存检索结果到全局 dict               │
│     │   → _emit("ambiguous") → 前端弹出按钮     │
│     │   → 用户点击后 selected_diagnosis 传入    │
│     │   → 命中缓存，跳过检索，直接生成报告      │
│     ├ 注入 LTM 偏好 + Entity 上下文            │
│     └ LLM 结构化提取 → 报告 JSON                │
│                                                │
│  4. 检测 _is_final: true                            │
│     ├ 保存 last_report（确保下一轮编辑可用）        │
│     ├ 跳过二次 LLM 调用，避免引入幻觉               │
│     └ 直接 json_to_display → _emit("report")         │
│                                                │
│  LLM 其他决策：                                 │
│  → 无 tool_calls → 直接流式回复（闲聊）        │
│  → edit_report → 修改已有报告，同样 is_final   │
│  → refine_report → 重写报告，同样 is_final     │
│  → 只有非 final 结果（错误等）才启用二次 LLM    │
└──────────────┬──────────────────────────────────┘
               ▼
┌─ Phase 3: 后处理（仅非 final 路径） ─────────────┐
│ 1. 更新 last_report（从工具结果提取）          │
│ 2. STM 记录对话轮次                            │
│ 3. _emit("message") → 前端渲染                  │
└────────────────────────────────────────────────┘
```

## 多轮对话机制

### 三层记忆架构

| 层级            | 模块                | 存储内容                       | 生命周期             |
| --------------- | ------------------- | ------------------------------ | -------------------- |
| Entity Tracker  | `entity_tracker.py` | 当前模态/部位/病史/诊断槽位    | 单次会话             |
| STM（短期记忆） | `short_term.py`     | 最近 N 轮对话历史 + 自动摘要   | 单次会话，可配置轮数 |
| LTM（长期记忆） | `long_term.py`      | 用户偏好（模态/部位/格式偏好） | 跨会话持久化         |

### 实体提取（Entity Tracker）

| 机制             | 说明                                                                                                                                                                               |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实体槽位         | `modality`（检查类型）、`body_part`（检查部位，**列表存储，支持多个**）、`clinical_history`（病史/症状）、`diagnosis`（已确认诊断列表）、`intent`（new_session / append / switch） |
| LLM 提取（优先） | 调用 LLM 返回 JSON `{"modality": "CT", "body_part": ["脑部", "颈部"]}`，支持数组输出多部位                                                                                         |
| 规则兜底         | LLM 不可用时按关键词长度降序匹配（PET-CT > CT > T），所有匹配部位追加到列表，长词优先去重（"膝关节"已匹配则跳过"膝"）                                                              |
| 上下文消解       | 省略主语时自动补全："再看看肝脏" → "CT 脑部 肝脏 再看看肝脏"（多部位拼接）                                                                                                         |
| 意图检测         | 粗粒度判断：new_session / append / switch（切换时清空 STM + last_report）                                                                                                          |

### Tool Calling 机制

| 工具              | 触发条件                             | 系统动作                                                             |
| ----------------- | ------------------------------------ | -------------------------------------------------------------------- |
| **rag_search**    | 生成/检索报告（有检查类型和部位）    | 多路召回 → Rerank 精排 → LLM 结构化提取 → 标记 `_is_final: true`     |
| **edit_report**   | 修改具体医学内容（如改CT值、删病变） | 调用 LLM 修改报告，保留 key 结构不破坏，标记 `_is_final: true`       |
| **refine_report** | 调整风格/详细程度（如"写详细点"）    | **跳过检索**，基于已有报告重写，保留医学内容，标记 `_is_final: true` |
| **直接回复**      | 闲聊/问候/功能咨询                   | LLM 直接回复，不调用工具                                             |

### 工具注册与执行

```
ToolRegistry 初始化
├── rag_search      ← create_rag_search_handler(chat_fn, ltm, entity_tracker, ..., last_report)
│                     内部使用 PromptBuilder.build("structure", ...) 统一拼装 System Prompt
├── edit_report     ← create_edit_report_handler(chat_fn, ..., last_report)
└── refine_report   ← create_refine_report_handler(chat_fn, ltm, entity_tracker, ..., last_report)
                      内部使用 PromptBuilder.build("refine", ...) 统一拼装 System Prompt

LLM 返回 tool_calls → registry.execute(id, name, arguments)
→ 返回 ToolResult(content, is_final)
→ is_final=True 时跳过二次 LLM 调用，直接发送报告
失败时返回 error JSON → LLM 可据此重试或向用户解释
```

### 对话流控制

| 机制       | 说明                                                           |
| ---------- | -------------------------------------------------------------- |
| 上下文消解 | 缺少检查类型时自动继承上一轮（"脑梗"→"CT 脑梗"）               |
| 指代消解   | "它"、"这个" 等代词替换为上一轮实体                            |
| 报告合并   | 新病变追加到已有报告，已有病变保留不覆盖                       |
| 短期记忆   | 可配置轮数（默认 10 轮），超出自动压缩为摘要                   |
| 切换清洗   | 意图切换时强制清空 STM 和 last_report，杜绝旧病灶残留          |
| 编辑保护   | EDIT 模式下保留报告 JSON 的 key 结构，仅按指令修改内容         |
| 重写保护   | REFINE 模式下 Key 集合变化时自动回退旧报告，杜绝医学内容被篡改 |

### 记忆检索注入

Phase 3 改动：改变全量注入 LTM/STM 到 Prompt 的做法，改为 **基于语义相关性按需检索**，仅将最相关的偏好和历史注入。

| 步骤               | 说明                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| 索引 LTM 偏好      | `MemoryRetriever.index_ltm()` → 增量追加，仅 Embedding 新偏好        |
| 索引 STM 历史      | `MemoryRetriever.index_stm()` → 增量追加，仅 Embedding 新增消息      |
| 检索               | 增强 Query → 向量化 → 余弦相似度排序 → 返回 Top-K（LTM: 3 / STM: 3） |
| PromptBuilder 拼装 | 只注入检索结果，减少无关记忆干扰，降低 Token 消耗                    |

| 配置         | 默认值 | 说明               |
| ------------ | ------ | ------------------ |
| `top_k_ltm`  | 3      | 返回的偏好条数     |
| `top_k_stm`  | 3      | 返回的历史消息条数 |
| `相似度阈值` | 0.3    | 低于阈值不返回结果 |

**增量索引策略**：首次调用时全量 Embedding，后续每轮仅对新出现的偏好/消息执行 Embedding。旧向量缓存复用，性能提升 ~90%（以 10 轮对话为例，API 调用从 ~400 次降至 ~50 次）。

## 工具设计与风险控制

在医疗报告这个严肃场景，我们对三个关键风险做了专门设计：

| 风险点                       | 解决方案                                                                                                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **二次 LLM 调用引入幻觉**    | 工具生成的结构化报告标记 `_is_final: true`，主循环检测后**跳过二次 LLM 调用**，直接将工具结果发送给前端。同时在 return 前保存 `last_report`，确保下一轮编辑/重写可以正常使用。                                                                   |
| **工具执行过程中前端无反馈** | 工具内部通过 `_emit("status")` 推送进度事件，每个阶段都有状态更新：<br>`searching` → `generating` → `done`，前端可以实时展示 Agent 思考过程。                                                                                                    |
| **LLM 漏传工具参数**         | 关键参数缺失时自动从记忆补全：<br>- `rag_search` 缺失 `modality`/`body_part` → 从 `EntityTracker` 补全<br>- `edit_report`/`refine_report` 缺失 `current_report` → 从 `last_report` 补全<br>工具层和主循环双层兜底，即使 LLM 漏传参数也不会报错。 |

### 歧义检测与用户选择

当知识库中存在同一疾病的多种变体（如"脑出血"、"基底节区脑出血"、"脑出血（破入脑室）"等），且 Rerank 分数接近时，系统自动触发歧义检测，让用户精确选择，而非由 LLM 猜测。

**歧义检测算法** (`detect_ambiguity` in `rag_tool.py`)：

```
1. Rerank top-10 中，如果 top-1 和 top-2 分数差 < 阈值（默认 0.03）
2. 收集 top-10 内所有不同的诊断结论
3. 如果不同诊断 ≥ 2 种 → 触发歧义
```

**交互流程**（支持多次点击不同按钮）：

```
第一轮：用户输入 "CT 脑出血"
  ├→ 多路召回 + Rerank（top-10）
  ├→ detect_ambiguity() → 检测到 9 种不同诊断
  ├→ 缓存 (search_result, reranked_entities) 到全局 dict
  ├→ _emit("ambiguous") → 前端渲染 9 个按钮
  └→ 等待用户选择

用户点击按钮 "基底节区脑出血"
  ├→ 前端传入 selected_diagnosis="基底节区脑出血"
  ├→ 跳过 Phase 1（实体提取/意图检测），保留 modality/body_part 槽位
  ├→ LLM 强制调用 rag_search（enhanced 注入明确指令）
  ├→ 命中缓存 → 跳过检索和 Rerank
  ├→ _filter_by_selected_diagnosis() → 只保留诊断结论匹配的参考
  ├→ _rebuild_search_result() → 重建检索文本，LLM 只看到匹配结果
  ├→ 生成对应报告 → _emit("report")
  └→ 缓存不删除 → 用户可以继续点击其他按钮

用户再次点击 "脑出血（破入脑室）"
  ├→ 同上流程 → 缓存命中 → 过滤 → 生成不同报告
  └→ 缓存仍然保留

用户输入新查询 "CT 脑梗" → 清除旧缓存 → 重新检索
```

**缓存加速**：

| 节点         | 第一轮  | 后续点击（任意次数） |
| ------------ | ------- | -------------------- |
| 多路召回     | ✅ 执行 | ❌ 跳过（缓存命中）  |
| Rerank       | ✅ 执行 | ❌ 跳过（缓存命中）  |
| 歧义检测     | ✅ 触发 | ❌ 跳过              |
| 结果过滤     | ❌      | ✅ 精确过滤匹配      |
| LLM 生成报告 | ❌      | ✅ 执行              |

**缓存生命周期**：全局 `_ambiguity_cache` dict 按 `session_id` 索引，**直到用户输入新查询才清除**。点击不同按钮不清除缓存，允许多次点击。

**过滤机制** (`_filter_by_selected_diagnosis` in `rag_tool.py`)：

```
精确匹配：d == selected_diagnosis → 优先返回这些
↓
如果没有精确匹配 → 返回包含匹配：(selected_diagnosis in d) or (d in selected_diagnosis)
↓
都没有 → 返回全部兜底
```

这样保证：用户点哪个 → LLM 只看到哪个 → 一定生成对应报告。

**实体槽位保留**：

当 `selected_diagnosis` 存在时，**跳过 Phase 1（实体提取/意图检测/上下文消解）**，保留上一轮的 `modality` 和 `body_part` 槽位。这样：

- 不会丢失检查类型/部位信息
- 缓存可以正确命中（参数完整）
- 避免重新提取清空原有槽位导致检索参数缺失

### 前端歧义交互

前端监听 `type: "ambiguous"` SSE 事件：

```javascript
case "ambiguous":
  addAmbiguousOptions(event)  // 渲染按钮列表，显示分数百分比
  break
```

用户点击按钮时，调用 `window.selectAmbiguityOption(option)`，将选项名作为 `selected_diagnosis` 参数传给后端 `/api/chat`。按钮样式仿照聊天界面，每行显示诊断名称和 Rerank 分数百分比。

### PromptBuilder 统一拼装

`rag_tool.py`、`refine_tool.py` 和 `rag_chat_v2.py` 中原本各自包含一套手动拼接 LTM 偏好 + Entity 上下文 + 模板 + Last Report 的代码，维护成本高且容易不一致。引入 `PromptBuilder` 后：

| 调用方           | `PromptBuilder.build()` 调用                                                           | 模板                   |
| ---------------- | -------------------------------------------------------------------------------------- | ---------------------- |
| `rag_tool.py`    | `PromptBuilder.build("structure", ltm_prefs=..., entity_context=..., last_report=...)` | `structure.md`         |
| `refine_tool.py` | `PromptBuilder.build("refine", ltm_prefs=..., entity_context=...)`                     | `refine.md`            |
| `rag_chat_v2.py` | `PromptBuilder.build("tool_orchestrator", ltm_prefs=..., entity_context=...)`          | `tool_orchestrator.md` |

拼接顺序（优先级从高到低）：LTM 偏好 → Entity 上下文 → 基础模板 → Last Report（自动去除 `reasoning` 字段），各段用 `\n\n---\n\n` 分隔。

### 检索前部位补全

`search_reports()` 在 `parse_query_keywords()` 提取关键词后，对缺失的部位做两层补全，增强元数据过滤和关键词检索路径的精确度。

**典型场景**：用户输入 "CT 脑出血"

- `parse_query_keywords()` 提取：检查类型="CT"，部位=""（空），诊断关键词=["出血"]
- 原因："脑"后面紧跟"出血"，不是独立部位词，不被 `_is_standalone_part()` 判定为部位

**两层补全机制**：

| 层级 | 来源                    | 说明                                                                                                                                                        |
| ---- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 层 1 | Entity Tracker 实体槽位 | 如果 `slots["body_part"]` 有值（LLM 提取或规则匹配得到），取第一个部位作为补全                                                                              |
| 层 2 | 诊断关键词启发式推断    | 如果 entity_tracker 也识别不到，通过 `_DIAGNOSIS_TO_PART` 映射表从诊断关键词或原始 query 中推断（如 "脑出血" → "头颅"、"肺结节" → "胸部"、"肝脏" → "腹部"） |

**设计原则**：只增强结构化过滤路径（元数据过滤 / 关键词检索）的部位字段，**不修改向量检索用的 query 文本**，保证语义相似度计算不受影响。

**效果对比**：

| 查询        | 补前 keywords            | 补后 keywords                | 对检索精度的影响                                           |
| ----------- | ------------------------ | ---------------------------- | ---------------------------------------------------------- |
| "CT 脑出血" | {检查类型:"CT", 部位:""} | {检查类型:"CT", 部位:"头颅"} | 元数据过滤从单条件变为双条件，排除其他部位的"出血"相关记录 |
| "CT 肺结节" | {检查类型:"CT", 部位:""} | {检查类型:"CT", 部位:"胸部"} | 关键词检索时增加部位过滤，排除腹部/头部的"结节"匹配        |

**实现位置**：`chat/rag_chat_v2.py` 中的 `_infer_part_from_diagnosis()` 函数和 `search_reports()` 的部位补全逻辑。

### 工具参数 Schema

| 工具              | 参数             | 必填 | 说明                                                                          |
| ----------------- | ---------------- | ---- | ----------------------------------------------------------------------------- |
| **rag_search**    | `query`          | 是   | 经过上下文消解的查询语句，包含检查类型和部位                                  |
|                   | `modality`       | 否   | 检查类型（CT/MR/MRI/DR/X线/超声/PET-CT/DSA/CTA/不限），缺失自动从 Entity 补全 |
|                   | `body_part`      | 否   | 检查部位，缺失自动从 Entity 补全                                              |
| **edit_report**   | `current_report` | 是   | 当前报告的 JSON 字符串，缺失自动从 last_report 补全                           |
|                   | `instruction`    | 是   | 修改指令，描述要改什么内容                                                    |
| **refine_report** | `current_report` | 是   | 当前报告的 JSON 字符串，缺失自动从 last_report 补全                           |
|                   | `style`          | 是   | 重写风格指令（如："更详细"、"简洁一点"）                                      |

## 向量数据库 Schema

集合名：`report_slices`（Milvus Lite）

| 字段       | 类型               | 说明             |
| ---------- | ------------------ | ---------------- |
| `id`       | INT64              | 主键，自增       |
| `vector`   | FLOAT_VECTOR(1024) | bge-m3 向量      |
| `text`     | VARCHAR(4096)      | 切片自然语言文本 |
| `source`   | VARCHAR(512)       | 来源文件名       |
| `检查类型` | VARCHAR(256)       | 如 CT、MRI       |
| `部位`     | VARCHAR(256)       | 检查部位         |
| `检查项目` | VARCHAR(256)       | 检查项目名称     |
| `诊断结论` | VARCHAR(1024)      | 诊断结论         |

索引：IVF_FLAT，COSINE 相似度

## 离线工具脚本

### 构建向量数据库

```bash
python app/data_pipeline/build_vector_db.py                           # 增量模式
python app/data_pipeline/build_vector_db.py --rebuild                 # 全量重建
python app/data_pipeline/build_vector_db.py --input ./xlsx_slices --batch-size 8
```

### 提取标准术语

```bash
python app/data_pipeline/extract_metadata.py
```

### 检索评估

```bash
python app/test/eval_retrieval.py                     # 抽 100 条，评估 top-1/3/5
python app/test/eval_retrieval.py -n 50 -k 5          # 抽 50 条，评估 top-5
```

### Rerank 对比评估

```bash
python app/test/eval_rerank_compare.py                     # 抽 30 条，向量 top-5，Rerank top-3
python app/test/eval_rerank_compare.py -n 50 -k 10 -r 5    # 抽 50 条，向量 top-10，Rerank top-5
```

## 技术栈

| 组件          | 技术                    | 说明                       |
| ------------- | ----------------------- | -------------------------- |
| Web 框架      | FastAPI + uvicorn       | SSE 流式推送               |
| 前端          | 原生 HTML/CSS/JS        | 无框架依赖                 |
| 向量数据库    | Milvus Lite (pymilvus)  | 轻量级本地向量库           |
| Embedding     | bge-m3 (1024 维)        | 本地部署，OpenAI 兼容 API  |
| Rerank        | Qwen3-VL-Reranker-8B    | SiliconFlow 云端 API       |
| LLM 生成/改写 | Qwen (qwen36-27b)       | 本地部署，OpenAI 兼容 API  |
| 配置管理      | config.yml + PyYAML     | 环境变量自动覆盖           |
| 短期记忆      | 内存 OrderedDict        | 轮次淘汰 + 自动摘要        |
| 会话持久化    | SQLite (Python sqlite3) | 会话数据自动保存，断线恢复 |
| 向量计算      | NumPy                   | 余弦相似度排序             |

## 会话持久化

`SessionStore` 将会话数据从内存迁移到 SQLite 数据库（`data/sessions.db`），实现断线恢复和跨进程会话管理。

### 数据库表结构

| 表              | 字段                                                                                       | 说明                   |
| --------------- | ------------------------------------------------------------------------------------------ | ---------------------- |
| `sessions`      | `id` (PK), `title`, `created_at`                                                           | 会话元数据             |
| `turns`         | `id` (PK), `session_id` (FK), `turn_index`, `user_input`, `assistant_output`, `created_at` | 对话记录，外键级联删除 |
| `session_state` | `session_id` (UNIQUE, FK), `entity_slots` (JSON), `last_report`, `updated_at`              | 实体槽位 + 上一轮报告  |

### 自动保存时机

| 时机           | 保存内容                                            |
| -------------- | --------------------------------------------------- |
| 每轮对话完成后 | `save_turn()` — 用户输入 + 助手输出                 |
| 每轮对话完成后 | `save_state()` — entity_tracker.slots + last_report |
| 首轮对话       | 自动更新标题（取用户输入前 20 字）                  |
| 会话创建时     | `create_session()` — 插入 sessions + session_state  |
| 会话删除时     | `delete_session()` — 级联删除三张表                 |

### 新增 API

| 接口                       | 方法   | 说明                    |
| -------------------------- | ------ | ----------------------- |
| `/api/sessions`            | GET    | 返回历史会话列表        |
| `/api/session?session_id=` | DELETE | 删除指定会话（级联）    |
| `/api/session/new`         | POST   | 生成并返回新 session_id |

### 容错设计

- 写入失败仅记录 Warning 日志，不阻断 Agent 主流程
- `load_session()` 返回 `None` 时自动创建新会话，兼容旧数据
- JSON 反序列化失败时回退为空字典 `{}`，不崩溃

## 前端对话存储（localStorage）

前端使用浏览器 localStorage 存储对话的完整 HTML（含思考过程），实现**刷新页面后无需调 API 即可恢复 UI**。

| 存储键                | 说明                                                                   |
| --------------------- | ---------------------------------------------------------------------- |
| `chatHistory`         | 对话列表（`[{id, title, time}, ...]`），用于左侧栏展示                 |
| `chatMessages_conv_*` | 单个对话的完整消息（`[{role, content, thinking}, ...]`），含 HTML 片段 |

### 保存时机

- 每轮 SSE 流结束后（`done` 事件）→ `saveCurrentConversation()`
- 将 assistant 消息的 `.thinking-container` 完整 HTML 保存到 `thinking` 字段
- 用户切换到其他对话前 → 先保存当前对话

### 刷新恢复流程

```
页面刷新 → initChatHistory()
  → 读取 localStorage → 渲染对话列表
  → 用户点击某个对话 → loadConversation(convId)
    → 读取 chatMessages_conv_xxx
    → 遍历消息：assistant 消息前插入 thinking-container（思考过程）
    → 渲染消息内容到 chatContainer
    → 不调任何后端 API，纯前端渲染
```

### 限制

- localStorage 是浏览器本地存储，**换浏览器/换电脑/清缓存后消失**
- 后端 SQLite（`data/sessions.db`）是真正的永久存储
- 两者互补：localStorage 负责 UI 快照恢复，SQLite 负责跨设备/跨会话恢复
