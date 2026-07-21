FROM python:3.11-slim

LABEL maintainer="report-generation"
LABEL description="Medical imaging report RAG generation system"

WORKDIR /app

# 使用阿里云镜像加速（国内部署）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY front/ ./front/
COPY python_start.py .
COPY config.yml .

# 创建数据目录
RUN mkdir -p data app/memory/data

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "python_start.py"]
