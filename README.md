# 医疗影像报告生成系统

基于 RAG（检索增强生成）的医疗影像报告智能生成系统。用户上传 xlsx 格式的影像报告模板，系统自动切片、向量化并入库，随后输入检查信息，系统检索相关报告切片，由 LLM 整合生成规范化的影像学表现与诊断意见。

详细文档见 [app/README.md](app/README.md)。

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
| `CHAT_MODEL`          | 生成模型名称         | `qwen36_27b_lora`                              |
| `RERANK_URL`          | Rerank API 地址      | `https://api.siliconflow.cn/v1/rerank`         |
| `RERANK_MODEL`        | Rerank 模型名称      | `Qwen/Qwen3-VL-Reranker-8B`                    |
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥 | （需填入）                                     |
| `HOST_PORT`           | 宿主机映射端口       | `7860`                                         |

### 2. 一键启动

```bash
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
