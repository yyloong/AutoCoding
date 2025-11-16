# MS-Agent 快速启动指南（省流版）

## 1️⃣ 克隆项目

```bash
git clone https://github.com/yyloong/AutoCoding.git
cd AutoCoding
```

## 2️⃣ 环境配置

### 系统要求
- Python >= 3.10（推荐 3.11+）
- Node.js 16+ & npm
- Git 和 Git LFS（用于 PaperBench 数据）

### 安装 Python 依赖
```bash
pip install -r requirements.txt
```

## 3️⃣ 获取 API Key（⭐ 重要）

### DashScope OpenAI 兼容接口
1. 访问阿里云 BaiLian：[https://bailian.console.aliyun.com/#/home](https://bailian.console.aliyun.com/#/home)
2. 登录或注册账户
3. 在左侧菜单找到"API 密钥"（或类似选项）
4. 创建新的 API Key（复制完整的 sk-xxxx 密钥）

### 设置环境变量

修改以下三个文件中的 `OPENAI_API_KEY` 为你的 API Key：
- `projects/code_scratch/architecture.yaml`
- `projects/code_scratch/coding.yaml`
- `projects/code_scratch/refine.yaml`


## 4️⃣ 运行 Code Scratch

### 方式 A：使用 CLI

```bash
python ms_agent/cli/cli.py run \
  --config projects/code_scratch \
  --query "请根据需求生成一个简单的前后端项目" \
  --trust_remote_code true
```

### 方式 B：使用快速启动脚本（Windows）

```powershell
cd projects\code_scratch
.\run_paperbench.bat debug code-dev
```

### 代码会生成到
```
output/
  ├── frontend/      # React + Vite 前端代码
  ├── backend/       # Node.js 后端代码
  └── files.json     # 生成的文件清单
```

## 5️⃣ 运行 PaperBench 评测（可选，目前未完成）

### 设置数据目录环境变量
```powershell
# PowerShell
$Env:PAPERBENCH_DATA_DIR = "完整路径\frontier-evals\project\paperbench\data"
setx PAPERBENCH_DATA_DIR "完整路径\frontier-evals\project\paperbench\data"

# CMD / Bash
set PAPERBENCH_DATA_DIR=完整路径/frontier-evals/project/paperbench/data
export PAPERBENCH_DATA_DIR="完整路径/frontier-evals/project/paperbench/data"
```

### 运行评测

```bash
# 3 篇论文（快速测试）
python projects/code_scratch/evaluate_paperbench.py --split debug --type code-dev

# 10 篇论文（中等规模）
python projects/code_scratch/evaluate_paperbench.py --split mini --type code-dev

# 全部 23 篇论文（完整评测，耗时很长）
python projects/code_scratch/evaluate_paperbench.py --split full --type code-dev
```

### 结果位置
```
paperbench_results/
  └── YYYYMMDD_HHMMSS/
      ├── results_final.json   # 最终评测结果
      └── results_temp.json    # 中间结果
```

## ⚠️ 已知问题与注意事项

### 1. 运行时间过长
- **问题**：单篇论文评测耗时 10-30 分钟（取决于论文复杂度和网络）
  - 原因：ms-agent 需要多轮 LLM 调用（architecture → coding → refine），每轮都是网络请求
  - DashScope API 响应可能较慢
- **建议**：
  - 首次测试用 `--split debug`（3 篇）
  - 不要并行多个评测进程，会导致 API 调用冲突或超时
  - 如果长时间无进度，检查网络连接和 API Key 是否有效

### 2. Windows GBK 编码问题
- **问题**：命令行输出可能出现乱码（Windows 10 系统）

### 2. 输出格式问题
- **问题**：`output/files.json` 靠我的直觉，输出格式可能不符合要求。考虑到paperbench的难度（openai测到的成功率最高26%），用弱模型大概率一个都通过不了。

## 📋 文件清单

| 文件 | 说明 |
|------|------|
| `projects/code_scratch/workflow.yaml` | 工作流配置（architecture → coding → refine） |
| `projects/code_scratch/coding.yaml` | 编码阶段 LLM 配置与 prompt |
| `projects/code_scratch/refine.yaml` | 调试阶段 LLM 配置与 prompt |
| `projects/code_scratch/callbacks/eval_callback.py` | 编译验证回调（已修复 Windows 兼容性） |
| `projects/code_scratch/evaluate_paperbench.py` | PaperBench 评测脚本 |
| `projects/code_scratch/run_paperbench.bat` | Windows 快速启动脚本 |

## 🔗 相关链接

- **MS-Agent 官方**：https://github.com/modelscope/ms-agent
- **AutoCoding 项目**：https://github.com/yyloong/AutoCoding
- **DashScope 控制台**：https://bailian.console.aliyun.com/#/home
- **PaperBench 官方**：https://github.com/openai/frontier-evals
- **Frontier Evals 数据**：https://github.com/openai/frontier-evals/tree/main/project/paperbench

## 💡 快速命令速查

```bash
# 设置 API Key（PowerShell）
$Env:OPENAI_API_KEY = "your_api_key"

# 设置 PaperBench 数据目录
$Env:PAPERBENCH_DATA_DIR = "path/to/frontier-evals/project/paperbench/data"

# 运行 code_scratch 短测试
python ms_agent/cli/cli.py run --config projects/code_scratch --query "简单测试：回复 Hello" --trust_remote_code true

# 运行 PaperBench debug 评测（3 篇，约 30-60 分钟）
python projects/code_scratch/evaluate_paperbench.py --split debug --type code-dev

# 检查最新结果
Get-Content .\paperbench_results\*\results_final.json | ConvertFrom-Json
```

---

**最后更新**：2025-11-16  
**当前状态**：功能完整，已知超时问题，暂未做性能优化
