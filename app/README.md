# 医疗影像报告生成Agent v2

基于 RAG（检索增强生成）的医学影像报告结构化输出系统，通过**意图识别 → 检索召回 → 结构化生成 → 多轮累积**的流水线，将用户输入的检查信息转化为规范的影像学报告。

## 核心特性

- **三段式工作流**：意图识别（SEARCH/EDIT/CHAT）→ 意图分叉 → 执行器，职责清晰
- **多轮对话**：上下文消解 + 指代消解，继承检查类型，多轮报告自动合并去重
- **短期记忆**：可配置轮数（前端配置）+ 实体追踪 + 历史摘要，可通过前端面板查看
- **Agent 思考过程**：前端可视化展示意图识别、查询改写、召回、Rerank、推理全流程
- **流式输出**：FastAPI + SSE 推送，报告和闲聊消息实时流式渲染
- **配置统一管理**：`config.yml` 一站式配置，环境变量自动覆盖
- **Rerank 降级**：Rerank 服务不可用时自动降级为纯向量检索

## 系统架构

```
用户输入 "CT脑出血"
       │
       ▼
┌──────────────────────────┐
│  预处理                   │
│  ├ 模糊检测 → 追问       │
│  ├ 上下文消解 → 指代继承  │
│  ├ 术语标准化             │
│  └ 查询改写               │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Stage 1: 意图识别（LLM） │
│  → SEARCH / EDIT / CHAT  │
└──────────┬───────────────┘
           │
   ┌───────┼───────┐
   ▼       ▼       ▼
 CHAT    EDIT   SEARCH
   │       │       │
   ▼       ▼       ▼
┌──────┐ ┌────┐ ┌────────────────────────┐
│ 闲聊  │ │编辑│ │ 多路召回（3路并行）      │
│ 回复  │ │已有│ │ ├ 向量检索（语义匹配）    │
│      │ │报告│ │ ├ 元数据过滤（精确匹配）   │
│      │ │    │ │ └ 关键词检索（全文匹配）   │
│      │ │    │ │ → 合并去重 → Rerank 精排 │
│      │ │    │ │ → LLM 结构化提取         │
│      │ │    │ │ → 合并新旧报告（去重）    │
└──┬───┘ └─┬──┘ └──────────┬─────────────┘
   │       │               │
   ▼       ▼               ▼
┌──────────────────────────┐
│  存储                     │
│  ├ 短期记忆 add_turn     │
│  └ last_report 更新      │
└──────────────────────────┘
```

## 项目结构

```
app/
├── chat/
│   ├── rag_chat_v2.py     # ★ 主入口：三段式工作流 + FastAPI Web 服务
│   ├── rag_chat.py        # 旧版（保留兼容）
│   ├── chat.py            # 命令行版
│   └── README.md          # chat 模块说明
├── rag/
│   ├── retrieval.py       # 多路召回模块（向量检索 + 元数据过滤 + 关键词检索）
│   ├── query_rewrite.py   # 查询改写模块（模糊检测 + 术语标准化 + 关键词解析 + LLM 改写）
│   ├── rerank.py          # Rerank 精排模块（SiliconFlow API）
│   └── prompt.md          # 系统提示词（影像报告结构化规范）
├── memory/
│   ├── short_term.py      # 短期记忆（对话历史 + 实体追踪 + 自动摘要）
│   ├── long_term.py       # 长期记忆（持久化存储）
│   └── README.md          # 记忆模块说明
├── data_pipeline/
│   ├── build_vector_db.py # 向量数据库构建（支持增量/全量重建）
│   ├── extract_metadata.py# 从 xlsx 提取标准术语（生成 metadata.json）
│   ├── xlsx_slicer.py     # xlsx 按行切片工具
│   ├── milvus_lite.db/    # Milvus Lite 向量数据库（运行时生成）
│   ├── xlsx_slices/       # 切片后的 Markdown 文件
│   └── report_template/   # 原始 xlsx 报告模板 + metadata.json
├── config.py              # 配置读取器（config.yml → 各模块）
├── test/                  # 测试与评估脚本
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

| 功能           | 说明                                                 |
| -------------- | ---------------------------------------------------- |
| 聊天输入       | 输入检查信息，如 `CT 脑出血`、`MR 膝关节`            |
| 结构化报告     | 影像学表现 + 诊断意见，实时流式渲染                  |
| 历史对话       | 左侧栏展示所有对话记录，支持切换、删除，对话自动保存 |
| 短期记忆       | 弹窗查看当前会话的对话历史、实体、摘要               |
| Agent 思考过程 | 折叠面板，展示意图识别、检索查询、召回、Rerank、推理 |
| 清空会话       | 重置会话状态和短期记忆                               |

### 命令行

```bash
python app/chat/rag_chat_v2.py
```

## 报告生成流程

完整流程分 6 个阶段：

```
用户输入 "脑梗"（第2轮，上轮是"CT 脑出血"）
       │
       ▼
┌─ 预处理 ────────────────────────────────────┐
│ 1. 模糊检测 → 不过于模糊，跳过              │
│ 2. 上下文消解 → "脑梗" 缺少检查类型 → 继承  │
│    上轮的 "CT" → "CT 脑梗"                  │
│ 3. 术语标准化 → "CT 脑梗"                   │
│ 4. 查询改写 → 有诊断词，无需改写             │
│ 5. 获取历史 → 最近 2 轮对话                 │
└──────────────┬──────────────────────────────┘
               ▼
┌─ Stage 1: 意图识别 ─────────────────────────┐
│ classify_intent("CT 脑梗", history)          │
│ → SEARCH（有诊断词，走检索）                 │
└──────────────┬──────────────────────────────┘
               ▼
┌─ Stage 3: 结构化提取 ───────────────────────┐
│ 1. search_reports() → 多路召回 + Rerank     │
│ 2. structure_report() → LLM 提取 JSON       │
│    system prompt: STRUCTURE_PROMPT           │
│      + "## 已生成的报告"（上一轮脑出血报告） │
│      + 检索结果 top-3                        │
│ 3. 合并新旧报告（脑出血 + 脑梗，key 去重）   │
│ 4. 更新 last_report + 短期记忆               │
│ 5. _emit("report") → 前端渲染               │
└──────────────────────────────────────────────┘
```

## 多轮对话机制

| 机制       | 说明                                             |
| ---------- | ------------------------------------------------ |
| 上下文消解 | 缺少检查类型时自动继承上一轮（"脑梗"→"CT 脑梗"） |
| 指代消解   | "它"、"这个" 等代词替换为上一轮实体              |
| 报告合并   | 新病变追加到已有报告，已有病变保留不覆盖         |
| 短期记忆   | 可配置轮数（默认 10 轮），超出自动压缩为摘要     |
| 实体追踪   | 从每轮报告中提取检查类型和病变名称               |

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

| 组件          | 技术                   | 说明                      |
| ------------- | ---------------------- | ------------------------- |
| Web 框架      | FastAPI + uvicorn      | SSE 流式推送              |
| 前端          | 原生 HTML/CSS/JS       | 无框架依赖                |
| 向量数据库    | Milvus Lite (pymilvus) | 轻量级本地向量库          |
| Embedding     | bge-m3 (1024 维)       | 本地部署，OpenAI 兼容 API |
| Rerank        | Qwen3-VL-Reranker-8B   | SiliconFlow 云端 API      |
| LLM 生成/改写 | Qwen (qwen36-27b)      | 本地部署，OpenAI 兼容 API |
| 配置管理      | config.yml + PyYAML    | 环境变量自动覆盖          |
| 短期记忆      | 内存 OrderedDict       | 轮次淘汰 + 自动摘要       |
