FROM python:3.11-slim

WORKDIR /app

# 安装必要系统工具（可选，可提高稳定性）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装（使用清华源加速，避免网络慢）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制整个翻译模块
COPY RepoTransAgent/ ./RepoTransAgent/

# 如果根目录有 projects_summary.jsonl，也复制（批量翻译用）
COPY projects_summary.jsonl ./

ENV PYTHONPATH=/app