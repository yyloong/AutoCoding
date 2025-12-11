FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NODE_ENV=production

# 使用清华镜像源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

# 安装系统依赖 + Python3.10 + pip
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl wget vim nano less tree htop \
    file unzip zip tar gzip \
    build-essential gcc g++ make cmake pkg-config \
    libssl-dev libffi-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev liblzma-dev \
    libncurses5-dev libncursesw5-dev \
    ca-certificates gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 统一 python/pip 命令
RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# 安装 Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# 为 pip 配置清华源（全局）
RUN printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn\n" > /etc/pip.conf

# 安装 Python 包
RUN pip install --upgrade pip setuptools wheel && \
    pip install pip-tools ipython jupyter numpy pandas black flake8 pytest

# 安装 Node.js 工具
RUN npm install -g yarn pnpm typescript ts-node nodemon

# 安装 uv（全局可用）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    chmod 755 /usr/local/bin/uv

WORKDIR /workspace

# （可选）安装一个 GPU 框架自检，比如 PyTorch
# RUN pip install --no-cache-dir "torch==2.3.0+cu121" -f https://download.pytorch.org/whl/torch_stable.html

# 验证安装（nvidia-smi 只在运行时有 GPU 时才会成功）
RUN python --version && pip --version && uv --version && \
    node --version && npm --version && \
    git --version && file --version && \
    echo "Setup complete!"

CMD ["/bin/bash"]