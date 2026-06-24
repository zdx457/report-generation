# 医疗影像报告生成系统

基于 RAG（检索增强生成）的医学影像报告结构化输出系统，通过术语标准化 + 查询改写 + 多路召回 + Rerank 精排 + LLM 生成，将用户输入的检查信息转化为规范的影像学报告。

## 系统架构

```
用户输入 "CT头部"
       │
       ▼
┌──────────────────────┐
│  模糊检测             │  过于模糊 → 追问用户
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  术语标准化 & 查询改写 │  "头部"→"头颅"（metadata.json）→ LLM 扩展查询词
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  多路召回（3路并行）   │
│  ├ 向量检索（语义匹配）│  改写文本 → bge-m3 → 1024维 → Milvus 余弦相似度
│  ├ 元数据过滤（精确）  │  标准化关键词 → 检查类型/部位/诊断结论精确匹配
│  └ 关键词检索（全文）  │  标准化关键词 → Milvus like + 检查类型/部位过滤
│  → 合并去重 → 候选集  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Rerank 精排          │  Cross-Encoder 交叉评分 → 带 Rerank 分数的 Top-N（精排）
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM 判断+生成        │  依据 Rerank 分数选最相关参考 → 结构化输出
└──────────────────────┘
```

> 📖 架构设计详解：[Rerank.md](markdown_files/Rerank.md) — 向量检索 vs Rerank 的原理、职责划分和 LLM 判断逻辑

## 核心功能

- **报告模板上传**：上传 xlsx 影像报告模板，自动按行切片为 Markdown 文件
- **向量化入库**：调用 bge-m3 模型将切片向量化，存入 Milvus Lite 向量数据库（支持增量入库）
- **模糊检测**：对过于简短的输入（如仅输入"CT"）进行追问，引导用户补充部位或诊断信息
- **术语标准化**：将用户输入的非标准术语映射为数据库标准术语（如"头部"→"头颅"），基于 metadata.json 动态加载
- **查询改写**：当用户输入不够具体时（如"CT头颅"），基于元数据标准术语通过 LLM 扩展为更完整的检索描述
- **多路召回**：向量检索 + 元数据过滤 + 关键词检索三路并行，合并去重后统一 Rerank 评分
- **Rerank 精排**：多路召回候选后，调用 Qwen3-VL-Reranker-8B 进行 Cross-Encoder 精排，返回带相关性分数的候选
- **LLM 判断生成**：LLM 依据 Rerank 分数选择最相关参考，进行结构化报告输出
- **检索过程可视化**：前端展示思考过程，包括查询改写、多路召回路径、Rerank 分数、检查类型等元信息
- **参数可调**：支持调整向量检索数量、Rerank 返回数量和生成温度
- **Rerank 降级**：Rerank 服务不可用时自动降级为纯向量检索结果

## 项目结构

```
app/
├── web.py               # Gradio Web 主程序
├── chat.py              # 命令行报告生成
├── retrieval.py         # 多路召回模块（向量检索 + 元数据过滤 + 关键词检索 + 合并去重）
├── query_rewrite.py     # 查询改写模块（模糊检测 + 术语标准化 + 关键词解析 + LLM 改写 + 追问）
├── extract_metadata.py  # 从 xlsx 模板提取标准术语（生成 metadata.json）
├── rerank.py            # Rerank 精排模块（独立封装）
├── prompt.md            # 系统提示词（影像报告整合规范）
├── build_vector_db.py   # 向量数据库构建脚本（支持增量/全量重建）
├── xlsx_slicer.py       # xlsx 按行切片工具
├── milvus_lite.db/      # Milvus Lite 向量数据库（运行时生成）
├── xlsx_slices/         # 切片后的 Markdown 文件
├── report_template/     # 原始 xlsx 报告模板
│   └── metadata.json    # 标准术语表（检查类型/部位/检查项目/诊断结论去重）
├── markdown_files/      # 详细文档
│   ├── Text_Embedding.md   # 数据入库与报告生成流程详解
│   └── Rerank.md           # Rerank 精排原理与 LLM 判断逻辑
└── test/                # 测试与评估脚本
    ├── _test_search.py              # 向量检索快速测试
    ├── eval_retrieval.py            # 检索命中率评估
    ├── eval_rerank_compare.py       # 向量检索 vs Rerank 对比评估
    ├── test_rerank_flow.py          # 完整流程测试（检索→Rerank→Prompt）
    └── tests_api/                   # API 连通性测试
        ├── deepsearcher_client_test.py
        ├── test_embedding.py
        ├── test_siliconflow_embedding.py
        └── call_qwen_model_test.py
```

## 环境变量配置

项目使用 `python-dotenv` 自动加载项目根目录下的 `.env` 文件，参考 `.env.example`：

```bash
# Embedding 服务（本地部署）
EMBED_URL=http://14.22.83.225:11002/v1/embeddings
EMBED_MODEL=bge-m3

# LLM 服务（本地部署）
CHAT_URL=http://14.22.86.97:11001/v1/chat/completions
CHAT_MODEL=qwen36_27b_lora

# Rerank 服务（SiliconFlow 云端 API）
RERANK_URL=https://api.siliconflow.cn/v1/rerank
RERANK_MODEL=Qwen/Qwen3-VL-Reranker-8B
SILICONFLOW_API_KEY=                     # 在 .env 中填写，不要提交到仓库
```

## 使用方式

### Web 界面

1. 打开浏览器访问 `http://服务器IP:7860`
2. 在右侧「上传报告模板」区域上传 xlsx 文件，点击「上传并处理」
3. 处理完成后，在左侧输入检查信息生成报告，例如：`CT弥漫性肺气肿的影像学表现`
4. 可调节参数：
   - **向量检索数量 (Top-K)**：从向量库中召回的候选数量（默认 5）
   - **Rerank 返回数量 (Top-K)**：Rerank 精排后返回给 LLM 的候选数量（默认 3）
   - **温度 (Temperature)**：LLM 生成温度（默认 0.7）

### 命令行

```bash
python app/chat.py                                    # 默认：向量检索 top-5，Rerank top-3
python app/chat.py --top-k=10 --rerank-top-k=3       # 向量检索 top-10，Rerank top-3
python app/chat.py --debug                            # 调试模式（打印检索和 Rerank 详情）
```

## 报告生成流程

用户输入 → 模糊检测 → 术语标准化 → 查询改写 → 多路召回 → Rerank 精排 → 构造 Prompt → 拼接提示词 → LLM 生成，七步完成：

```
用户输入 "CT头部"
       │
       ▼
① 用户输入 & 模糊检测
   → 过于模糊（如仅输入"CT"）→ 追问用户，终止流程
   → 不够具体（如"CT头颅"）→ 进入改写流程
   → 足够具体 → 直接使用原始查询
       │
       ▼
② 术语标准化 & 查询改写
   → 标准化：基于 metadata.json 映射非标准术语（"头部"→"头颅"）
   → 解析关键词：从标准化文本提取检查类型/部位/诊断（供路径2、3使用）
   → 改写：基于元数据标准术语，LLM 扩展为更完整的检索描述（供路径1使用）
       │
       ▼
③ 多路召回 top-K（每路独立检索，合并去重）
   ├ 路径1: 向量检索（改写文本 → bge-m3 + Milvus 余弦匹配）→ 语义相似度候选
   ├ 路径2: 元数据过滤（标准化关键词 → 检查类型/部位/诊断精确匹配）→ 精确匹配候选
   └ 路径3: 关键词检索（标准化关键词 → Milvus like + 检查类型/部位过滤）→ 全文匹配候选
   → 三路合并去重 → 候选集
       │
       ▼
④ Rerank 精排 top-N（Qwen3-VL-Reranker-8B 交叉评分）
   → 对候选集逐一与问题做 Cross-Encoder 评分
   → 按分数重新排序，取前 N 条
       │
       ▼
⑤ 构造 Prompt（拼接 N 条参考 + Rerank分数 + 用户问题）
   → LLM 收到带分数的参考信息
       │
       ▼
⑥ 拼接提示词（System Prompt + 参考信息 + 用户问题）
       │
       ▼
⑦ LLM 判断+生成
   → 依据 Rerank 分数选择最相关参考
   → 对最相关参考做结构化输出（影像学表现 + 诊断意见）
```

> 📖 完整流程详解：[Text_Embedding.md](markdown_files/Text_Embedding.md) — 数据入库6步 + 报告生成6步，含代码对应和示例

## 数据入库流程

上传 xlsx → 切片 → 增量检查 → 向量化 → 写入 Milvus，5步完成：

```
上传 xlsx 文件
    │
    ▼
① 保存到 report_template/ 目录
    │
    ▼
② 按行切片：每行 → 一个 Markdown 文件（保存到 xlsx_slices/）
    │
    ▼
③ 增量检查：对比 Milvus 已有 source，跳过已入库切片
    │
    ▼
④ bge-m3 批量向量化（batch_size=16，带5次重试）
    │
    ▼
⑤ 写入 Milvus Lite（向量 + 文本 + 元数据）
```

> 📖 入库流程详解：[Text_Embedding.md](markdown_files/Text_Embedding.md#一数据入库流程上传文件)

## 向量数据库 Schema

集合名：`report_slices`

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

### 提取标准术语

```bash
python app/extract_metadata.py                          # 从 report_template/ 中提取标准术语
```

> 生成 `app/report_template/metadata.json`，供术语标准化和查询改写使用。xlsx 模板更新后需重新运行。

### xlsx 切片

```bash
python app/xlsx_slicer.py --input ./app/report_template --output ./app/xlsx_slices
```

### 构建向量数据库

```bash
python app/build_vector_db.py                          # 增量模式
python app/build_vector_db.py --rebuild                # 全量重建
python app/build_vector_db.py --input ./app/xlsx_slices --batch-size 8
```

### 检索评估

```bash
python app/test/eval_retrieval.py                    # 默认抽 100 条，评估 top-1/3/5
python app/test/eval_retrieval.py -n 50 -k 5         # 抽 50 条，评估 top-5
```

### Rerank 对比评估

```bash
python app/test/eval_rerank_compare.py                    # 默认抽 30 条，向量 top-5，Rerank top-3
python app/test/eval_rerank_compare.py -n 50 -k 10 -r 5   # 抽 50 条，向量 top-10，Rerank top-5
python app/test/eval_rerank_compare.py --no-rerank         # 只评估向量检索
```

## 技术栈

| 组件       | 技术                            | 说明                      |
| ---------- | ------------------------------- | ------------------------- |
| Web 框架   | Gradio >= 4.0                   | 交互界面                  |
| 向量数据库 | Milvus Lite (pymilvus >= 2.4.0) | 轻量级本地向量库          |
| Embedding  | bge-m3 (1024 维)                | 本地部署，OpenAI 兼容 API |
| Rerank     | Qwen3-VL-Reranker-8B            | SiliconFlow 云端 API      |
| LLM        | Qwen (qwen36_27b_lora)          | 本地部署，OpenAI 兼容 API |
| 环境管理   | python-dotenv >= 1.0.0          | 自动加载 .env 配置        |
| 容器化     | Docker + Docker Compose         | 部署方案                  |

## 详细文档索引

| 文档                                                  | 内容                                                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [Text_Embedding.md](markdown_files/Text_Embedding.md) | 数据入库流程（6步详解）+ 报告生成流程（7步详解）+ 术语标准化 + 关键词解析 + 多路召回 vs Rerank 对比 |
| [Rerank.md](markdown_files/Rerank.md)                 | Rerank 精排原理 + LLM 判断逻辑 + rerank_top_k 参数含义 + 降级机制                                   |
