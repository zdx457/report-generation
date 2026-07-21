# 医疗影像报告生成Agent v2

基于 RAG（检索增强生成）的医疗影像报告智能生成系统。用户上传 Excel 格式的影像报告模板，系统自动切片、双通道向量化并入库。输入检查信息后，系统通过多路召回 + Rerank 检索相关报告，由 LLM 整合生成规范化的影像学表现与诊断意见。

## 核心特性

- **双通道向量化**：每个切片生成完整行向量和影像学表现向量，提升检索准确率
- **多路召回**：向量检索 + 元数据过滤 + 关键词匹配，三路互补
- **Rerank 重排**：使用 Rerank 模型对召回结果精排
- **歧义检测**：自动检测多种相关诊断，提供交互式选择
- **多轮对话**：支持上下文继承、意图切换、报告合并
- **会话持久化**：SQLite 存储，服务重启后对话不丢失
- **长期记忆**：自动学习用户偏好，个性化报告生成
- **流式输出**：实时显示生成进度和思考过程

## 架构

```
前端 (Vue.js)  ←→  FastAPI 后端  ←→  LLM (qwen36-27b)
                                    ←→  Milvus Lite (向量数据库)
                                    ←→  SQLite (会话/记忆持久化)
```

## 部署

### 前置条件

- Docker & Docker Compose 已安装并运行
- Embedding API 和 Chat API 服务可用

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入实际的 API 地址和模型名称
```

`.env` 配置项：

| 变量                  | 说明                 | 默认值                                         |
| --------------------- | -------------------- | ---------------------------------------------- |
| `EMBED_URL`           | Embedding API 地址   | `http://14.22.83.225:11002/v1/embeddings`      |
| `EMBED_MODEL`         | 向量化模型名称       | `bge-m3`                                       |
| `CHAT_URL`            | Chat API 地址        | `http://14.22.86.97:11001/v1/chat/completions` |
| `CHAT_MODEL`          | 生成模型名称         | `qwen36-27b`                                   |
| `RERANK_URL`          | Rerank API 地址      | `https://api.siliconflow.cn/v1/rerank`         |
| `RERANK_MODEL`        | Rerank 模型名称      | `Qwen/Qwen3-VL-Reranker-8B`                    |
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥 | （需填入）                                     |
| `HOST_PORT`           | 宿主机映射端口       | `7860`                                         |

### 2. 一键启动

```bash
sed -i 's/\r$//' start.sh
chmod +x start.sh
./start.sh start
```

启动后访问 `http://服务器IP:7860`

### 3. 管理命令

```bash
./start.sh start     # 初始化并启动服务
./start.sh stop      # 停止服务
./start.sh restart   # 重启服务
./start.sh rebuild   # 无缓存重新构建并启动
./start.sh logs      # 查看实时日志
./start.sh status    # 查看服务状态
./start.sh init      # 仅初始化（不启动）
```

## 本地开发

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python python_start.py
```

访问 `http://localhost:7860`

## 详细文档

- [后端架构](app/README.md)
- [记忆系统](app/memory/README.md)
- [工具调用](app/chat/README.md)
