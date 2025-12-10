FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NODE_ENV=production

# 安装系统依赖（包括常用文件工具）
RUN apt-get update && apt-get install -y \
    git curl wget vim nano less tree htop \
    file unzip zip tar gzip \
    build-essential gcc g++ make cmake pkg-config \
    libssl-dev libffi-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev liblzma-dev \
    libncurses5-dev libncursesw5-dev \
    ca-certificates gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 安装 Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 包
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    pip-tools ipython jupyter numpy pandas black flake8 pytest

# 安装 Node.js 工具
RUN npm install -g yarn pnpm typescript ts-node nodemon

# 安装 uv（全局可用）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    chmod 755 /usr/local/bin/uv

WORKDIR /workspace

# 验证安装
RUN python --version && pip --version && uv --version && \
    node --version && npm --version && \
    git --version && file --version && \
    echo "Setup complete!"

CMD ["/bin/bash"]