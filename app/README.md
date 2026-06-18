# 医疗影像报告生成系统 — 详细文档

## 系统架构

```
用户提问
   │
   ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Gradio Web  │────▶│  bge-m3 向量化  │────▶│  Milvus Lite     │
│  界面 (7860) │     │  Embedding API  │     │  向量检索         │
└──────────────┘     └─────────────────┘     └──────────────────┘
       │                                            │
       │              检索结果                        │
       │◀───────────────────────────────────────────┘
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
- **报告生成**：用户输入检查信息 → 向量检索召回相关报告切片 → LLM 整合生成规范化的影像学表现与诊断意见
- **检索过程可视化**：前端展示思考过程，包括检索来源、相似度、检查类型等元信息
- **参数可调**：支持调整检索数量（Top-K）和生成温度（Temperature）

## 项目结构

```
app/
├── web.py               # Gradio Web 主程序（容器入口）
├── chat.py              # 命令行报告生成
├── prompt.md            # 系统提示词（影像报告整合规范）
├── build_vector_db.py   # 向量数据库构建脚本（支持增量/全量重建）
├── xlsx_slicer.py       # xlsx 按行切片工具
├── eval_retrieval.py    # 检索命中率评估脚本
├── milvus_lite.db/      # Milvus Lite 向量数据库（运行时生成）
├── xlsx_slices/         # 切片后的 Markdown 文件
└── report_template/     # 原始 xlsx 报告模板
```

## 使用方式

### Web 界面

1. 打开浏览器访问 `http://服务器IP:7860`
2. 在右侧「上传报告模板」区域上传 xlsx 文件，点击「上传并处理」
3. 处理完成后，在左侧输入检查信息生成报告，例如：`CT弥漫性肺气肿的影像学表现`

### 命令行

```bash
# 进入容器
docker exec -it report-generation bash

# 命令行生成报告
python rag/chat.py
python rag/chat.py --top-k 5 --debug
```

## 离线工具脚本

以下脚本可在本地直接运行（无需 Docker），用于数据预处理和评估：

### xlsx 切片

```bash
python rag/xlsx_slicer.py --input ./rag/report_template --output ./rag/xlsx_slices
```

### 构建向量数据库

```bash
python rag/build_vector_db.py                          # 增量模式
python rag/build_vector_db.py --rebuild                # 全量重建
python rag/build_vector_db.py --input ./rag/xlsx_slices --batch-size 8
```

### 检索评估

```bash
python rag/eval_retrieval.py                    # 默认抽 100 条，评估 top-1/3/5
python rag/eval_retrieval.py -n 50 -k 5         # 抽 50 条，评估 top-5
python rag/eval_retrieval.py -n 200 -k 1 3 5    # 抽 200 条，同时评估 top-1/3/5
```

## 数据流程

```
xlsx 报告模板
    │
    ▼  xlsx_slicer.py / Web 上传
Markdown 切片（每行一个 .md 文件）
    │
    ▼  build_vector_db.py / Web 上传
Milvus Lite 向量数据库（bge-m3 1024 维）
    │
    ▼  web.py / chat.py
用户输入 → 向量检索 → LLM 生成报告
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

## 技术栈

- **Web 框架**：Gradio >= 4.0
- **向量数据库**：Milvus Lite（pymilvus[milvus_lite] >= 2.4.0）
- **Embedding**：bge-m3（1024 维，通过 OpenAI 兼容 API 调用）
- **LLM**：Qwen（通过 OpenAI 兼容 API 调用）
- **容器化**：Docker + Docker Compose
