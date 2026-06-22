# 医疗影像报告生成系统 — 详细文档

## 系统架构

```
用户提问
   │
   ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Gradio Web  │────▶│  bge-m3 向量化  │────▶│  Milvus Lite     │
│  界面 (7860) │     │  Embedding API  │     │  向量检索 Top-K   │
└──────────────┘     └─────────────────┘     └──────────────────┘
       │                                            │
       │              检索结果                        │
       │◀───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Qwen3-VL-Reranker   │────▶  按 Query-Doc 交叉注意力重排序，取 Top-N
│  Rerank 精排          │
└──────────────────────┘
       │
       ▼
┌─────────────────┐
│  Qwen LLM API   │────▶  整合后影像学表现 + 凝练诊断意见（报告生成）
│  生成报告        │
└─────────────────┘
```

## 核心功能

- **报告模板上传**：上传 xlsx 影像报告模板，自动按行切片为 Markdown 文件
- **向量化入库**：调用 bge-m3 模型将切片向量化，存入 Milvus Lite 向量数据库（支持增量入库）
- **Rerank 精排**：向量检索召回 Top-K 候选后，调用 Qwen3-VL-Reranker-8B 进行 Cross-Encoder 精排，选取最相关文档
- **报告生成**：用户输入检查信息 → 向量检索 → Rerank 精排 → LLM 整合生成规范化的影像学表现与诊断意见
- **检索过程可视化**：前端展示思考过程，包括向量检索结果、Rerank 分数、检查类型等元信息
- **参数可调**：支持调整向量检索数量（Top-K）、Rerank 返回数量（Top-K）和生成温度（Temperature）
- **Rerank 降级**：Rerank 服务不可用时自动降级为纯向量检索结果，不影响正常使用

## 项目结构

```
app/
├── web.py               # Gradio Web 主程序（容器入口）
├── chat.py              # 命令行报告生成（含 Rerank）
├── prompt.md            # 系统提示词（影像报告整合规范）
├── build_vector_db.py   # 向量数据库构建脚本（支持增量/全量重建）
├── xlsx_slicer.py       # xlsx 按行切片工具
├── eval_retrieval.py    # 检索命中率评估脚本
├── milvus_lite.db/      # Milvus Lite 向量数据库（运行时生成）
├── xlsx_slices/         # 切片后的 Markdown 文件
├── report_template/     # 原始 xlsx 报告模板
└── test/                # 测试脚本
    ├── _test_search.py              # 向量检索快速测试
    ├── test_rerank_flow.py          # 完整流程测试（检索→Rerank→Prompt）
    └── tests_api/                   # API 连通性测试
        ├── deepsearcher_client_test.py   # DeepSearcher API 测试
        ├── test_embedding.py             # 本地 Embedding API 测试
        ├── test_siliconflow_embedding.py # SiliconFlow Embedding API 测试
        ├── call_qwen_model_test.py      # Qwen LLM API 测试
        └── system_prompt.md             # 测试用系统提示词
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
   - **Rerank 返回数量 (Top-K)**：从候选中精选的数量（默认 1）
   - **温度 (Temperature)**：LLM 生成温度（默认 0.7）

### 命令行

```bash
python app/chat.py                                    # 默认：向量检索 top-5，Rerank top-1
python app/chat.py --top-k=10 --rerank-top-k=3       # 向量检索 top-10，Rerank top-3
python app/chat.py --debug                            # 调试模式（打印检索和 Rerank 详情）
```

## 离线工具脚本

以下脚本可在本地直接运行（无需 Docker），用于数据预处理和评估：

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
python app/eval_retrieval.py                    # 默认抽 100 条，评估 top-1/3/5
python app/eval_retrieval.py -n 50 -k 5         # 抽 50 条，评估 top-5
python app/eval_retrieval.py -n 200 -k 1 3 5    # 抽 200 条，同时评估 top-1/3/5
```

## 数据流程

详细的数据流程文档请查看 [Text_Embedding.md](markdown_files/Text_Embedding.md)，包含：

- **数据入库流程**：xlsx 上传 → 切片 → 增量检查 → 向量化 → 写入 Milvus（6步详解）
- **报告生成流程**：向量检索 → Rerank 精排 → 拼接 Prompt → LLM 生成（4步详解）
- **向量检索 vs Rerank 对比**：Bi-Encoder 与 Cross-Encoder 的原理和适用场景

简要流程：

```
xlsx 上传 → 按行切片 → 增量检查 → 向量化 → 写入 Milvus
                                                    │
用户问题 → bge-m3 向量化 → Milvus 检索 Top-K → Rerank 精排 Top-N → 拼接 Prompt → LLM 生成报告
```

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

## 向量文本构造方式

切片 md 文件解析后，将表格字段拼接为自然语言文本用于向量化：

```
检查类型：CT
部位：头颅
检查项目：颅脑平扫
诊断结论：脑出血
影像学表现：侧基底节区见团块状高密度影，CT值约61HU...
影像学意见：侧基底节区脑出血。
来源up_id：3000017
来源id：3000073
模态/部位/检查：CT / 头颅 / 颅脑平扫
```

其中 `检查类型`、`部位`、`检查项目`、`诊断结论` 同时作为元数据单独存储，支持过滤查询。

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
