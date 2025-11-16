# PaperBench 快速启动指南

## ⚡ 5 分钟快速开始

### 1️⃣ 准备环境（仅需一次）

```bash
# 1. 克隆 PaperBench 官方仓库
git clone https://github.com/openai/frontier-evals.git --filter=blob:none

# 2. 进入目录
cd frontier-evals

# 3. 下载数据集（使用 Git LFS）
git lfs fetch --include "project/paperbench/data/**"
git lfs checkout project/paperbench/data

# 4. 设置环境变量
export PAPERBENCH_DATA_DIR="$(pwd)/project/paperbench/data"
# 或添加到 ~/.bashrc (Linux/Mac) 或环境变量 (Windows)

# 5. 验证数据集
ls $PAPERBENCH_DATA_DIR/papers/ | head -5
```

### 2️⃣ 配置 API Keys

```bash
# 在 frontier-evals/project/paperbench 目录中
cd project/paperbench

# 编辑 .env 文件
cp .env.example .env
nano .env  # 或用你喜欢的编辑器

# 填入以下内容：
# OPENAI_API_KEY=sk-xxx...
# GRADER_OPENAI_API_KEY=sk-xxx...（可选，默认同上）
```

### 3️⃣ 运行快速评测（推荐）

```bash
# 进入 ms-agent 项目目录
cd /path/to/ms-agent

# 运行评测工具
python projects/code_scratch/evaluate_paperbench.py \
  --split debug \
  --type code-dev \
  --paperbench-dir $PAPERBENCH_DATA_DIR

# 输出示例：
# ✓ 评测器初始化成功
#   - PaperBench 数据目录: /path/to/data
#   - 评测类型: code-dev
#   - 论文分割: debug
#
# 🔍 检验环境配置...
#   ✓ PaperBench 数据目录: ✓ 已设置
#   ...
#
# 🚀 开始评测...
# [1/3] 评测论文: dpo-direct-preference
#   → 结果已保存: paperbench_results/20250116_120000/results_temp.json
# ...
#
# 📊 评测总结
# ================================================
# 总论文数:        3
# 完成:            3
# 失败:            0
# 成功率:          100.0%
# 平均分数:        0.50
# ================================================
```

---

## 📊 调试和提速

### 快速模式对比

| 模式 | 论文数 | 运行时间 | 成本 | 用途 |
|------|--------|---------|------|------|
| `--split debug` | 3 篇 | ~5-10 分钟 | 最低 | 测试和验证 |
| `--split mini` | 10 篇 | ~30-60 分钟 | 中等 | 快速评估 |
| `--split full` | 20 篇 | ~2-4 小时 | 高 | 完整评测 |

### 评测类型对比

| 类型 | 评估内容 | 需要 GPU | 成本 | 评分范围 |
|------|---------|---------|------|---------|
| `--type code-dev` | 仅代码质量 | ✗ 不需要 | 低 | 0-100 (代码分) |
| `--type complete` | 代码+执行+结果 | ✓ 需要 | 高 | 0-100 (综合分) |

### 推荐配置

```bash
# 快速验证（新手推荐）
python projects/code_scratch/evaluate_paperbench.py \
  --split debug --type code-dev

# 快速评估（2个选项）
python projects/code_scratch/evaluate_paperbench.py \
  --split mini --type code-dev

# 完整评估（需要 GPU）
python projects/code_scratch/evaluate_paperbench.py \
  --split full --type complete --is-gpu true
```

---

## 🔧 与 MS-Agent Code Scratch 集成（高级）

### 步骤 1: 修改 Code Scratch 配置

编辑 `projects/code_scratch/refine.yaml`：

```yaml
# 在文件末尾添加
paperbench:
  enabled: true
  eval_type: code-dev  # 或 complete
  data_dir: ${PAPERBENCH_DATA_DIR}
```

### 步骤 2: 为 PaperBench 优化 Prompt

编辑 `projects/code_scratch/architecture.yaml`，在 system prompt 中添加：

```yaml
prompt:
  system: |
    [原有的 system prompt...]
    
    # 特殊说明：如果这是一项学术论文复现任务：
    1. 仔细阅读并理解论文的：
       - 核心创新（Main Contributions）
       - 关键方法（Methodology）
       - 实验设置（Experimental Setup）
    
    2. 设计代码结构应该：
       - 实现论文中的所有关键算法
       - 支持论文中使用的数据集
       - 能复现论文中报告的结果
    
    3. 代码质量要求：
       - 清晰的模块划分
       - 完善的错误处理
       - 可复现的随机种子设置
```

### 步骤 3: 通过 MS-Agent 运行 PaperBench

```bash
# 使用 MS-Agent 评测单篇论文
cd /path/to/ms-agent

python ms_agent/cli/cli.py run \
  --config projects/code_scratch \
  --query "复现论文：DPO: Direct Preference Optimization。从理解论文开始，生成完整的实现代码。" \
  --trust_remote_code true
```

---

## 📈 查看结果

### 结果文件位置

```
paperbench_results/
├── 20250116_120000/          # 时间戳目录
│   ├── results_temp.json      # 中间结果
│   └── results_final.json     # 最终结果
├── 20250116_150000/
│   └── ...
```

### 解析结果

```bash
# 查看最新结果
cat paperbench_results/*/results_final.json | python -m json.tool

# 提取关键指标
python -c "
import json
with open('paperbench_results/*/results_final.json') as f:
    data = json.load(f)
    summary = data['summary']
    print(f'成功率: {summary[\"success_rate\"]:.1%}')
    print(f'平均分数: {summary[\"average_score\"]:.2f}')
    for paper in data['papers']:
        print(f'{paper[\"paper_id\"]}: {paper.get(\"score\", 0):.2f}')
"
```

---

## 🐛 常见问题

### Q: "找不到 PAPERBENCH_DATA_DIR"

```bash
# 解决方案 1: 临时设置
export PAPERBENCH_DATA_DIR=/path/to/frontier-evals/project/paperbench/data

# 解决方案 2: 永久设置（Linux/Mac）
echo 'export PAPERBENCH_DATA_DIR=/path/to/frontier-evals/project/paperbench/data' >> ~/.bashrc
source ~/.bashrc

# 解决方案 3: 永久设置（Windows PowerShell）
[Environment]::SetEnvironmentVariable("PAPERBENCH_DATA_DIR", "C:\path\to\paperbench\data", "User")
```

### Q: "API Key 无效"

```bash
# 检查 API Key 是否设置
echo $OPENAI_API_KEY  # Linux/Mac
echo $Env:OPENAI_API_KEY  # Windows PowerShell

# 检查 .env 文件
cat project/paperbench/.env | grep OPENAI_API_KEY
```

### Q: "找不到 papers 目录"

```bash
# 检查 Git LFS 是否正确安装
git lfs version

# 重新拉取数据
cd frontier-evals
git lfs fetch --include "project/paperbench/data/**" --force
git lfs checkout project/paperbench/data --force

# 验证数据
ls project/paperbench/data/papers/ | wc -l  # 应该显示 20
```

### Q: "评测超时或很慢"

```bash
# 使用 debug 模式（只测 3 篇）
--split debug

# 检查 API 速率限制
# 如果经常超时，可能是 API 限制，需要增加等待时间
```

### Q: GPU 相关错误

```bash
# 检查 GPU 是否可用
nvidia-smi

# 如果没有 GPU，使用 code-dev 模式（不需要 GPU）
--type code-dev
```

---

## 📚 深度学习资源

### 理解 PaperBench

1. **官方论文**：https://arxiv.org/abs/2504.01848
   - 详细的评估方法论
   - 20 篇 ICML 论文的特点分析

2. **GitHub 代码**：https://github.com/openai/frontier-evals/tree/main/project/paperbench
   - 完整的评估框架
   - 自定义 Agent 示例

### 查看论文评估标准

```bash
# 使用官方 Web 界面查看
cd frontier-evals/project/paperbench

# 启动 GUI（需要图形环境）
uv run python paperbench/gui/app.py \
  --path-to-paper ./data/papers/dpo-direct-preference \
  --rubric-file-name rubric.json
```

### 优化 Agent 性能

查看 `frontier-evals/project/paperbench/paperbench/agents/` 中的：
- `aisi-basic-agent/`: 基础 ReAct 智能体（推荐参考）
- `config.yaml`: Agent 配置示例

---

## 🎯 下一步

1. **验证设置**：运行 `--split debug` 确保一切工作正常
2. **优化提示词**：根据评分结果调整 MS-Agent 的 system prompt
3. **扩展测试**：从 `debug` 升级到 `mini`，再到 `full`
4. **分析薄弱点**：找出低分论文的共同特征
5. **迭代改进**：基于反馈持续优化代码生成能力

---

## 💡 预期结果

根据 PaperBench 官方基线：

| 模型 | Code-Dev 分数 | 完整分数 |
|------|-------------|--------|
| **Claude 3.5 Sonnet** | 21.0% | 16.1% |
| **o1-high (36h)** | 26.0% | 13.2% |
| **GPT-4o** | 4.1% | - |

**目标**：让 MS-Agent Code Scratch 在 Code-Dev 模式下超过 21% 的 Claude 3.5 Sonnet 基线。

---

**祝你评测顺利！** 🚀

有任何问题，欢迎查阅官方文档或联系 MS-Agent 社区。
