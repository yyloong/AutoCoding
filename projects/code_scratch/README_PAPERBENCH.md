# 🎯 MS-Agent Code Scratch - PaperBench 评测套件

这是一个**完整的 AI 论文复现能力评测框架**，用于评估 MS-Agent Code Scratch 在 OpenAI PaperBench 基准上的表现。

## 📚 文件说明

本目录包含以下文件：

### 📖 文档

| 文件 | 用途 | 阅读时间 |
|------|------|--------|
| **PAPERBENCH_QUICKSTART.md** | ⚡ 5分钟快速开始指南 | 5-10 分钟 |
| **PAPERBENCH_EVALUATION.md** | 📚 完整评测方案和理论 | 30-45 分钟 |
| **PAPERBENCH_SUMMARY.md** | 📋 项目总结和概览 | 10-15 分钟 |

### 🔧 工具和脚本

| 文件 | 用途 | 平台 |
|------|------|------|
| **evaluate_paperbench.py** | 🐍 一键评测脚本（推荐） | Windows/Linux/Mac |
| **run_paperbench.sh** | 🐧 Bash 快速启动脚本 | Linux/Mac |
| **run_paperbench.bat** | 🪟 Windows 快速启动脚本 | Windows |

---

## 🚀 快速开始（3 步）

### 1️⃣ 准备 PaperBench 数据

```bash
# 克隆官方仓库
git clone https://github.com/openai/frontier-evals.git --filter=blob:none
cd frontier-evals

# 下载 20 篇 ICML 2024 论文
git lfs fetch --include "project/paperbench/data/**"
git lfs checkout project/paperbench/data

# 设置环境变量
export PAPERBENCH_DATA_DIR="$(pwd)/project/paperbench/data"
```

### 2️⃣ 配置 API Keys

```bash
# 在 ms-agent 项目中
export OPENAI_API_KEY="sk-xxx..."
```

### 3️⃣ 运行评测

```bash
# 方式 A：使用 Python 脚本（推荐）
python projects/code_scratch/evaluate_paperbench.py \
  --split debug --type code-dev

# 方式 B：使用快速启动脚本（Linux/Mac）
bash projects/code_scratch/run_paperbench.sh debug code-dev

# 方式 C：使用 Windows 脚本
projects\code_scratch\run_paperbench.bat debug code-dev
```

**预期结果**：5-10 分钟内完成对 3 篇论文的评测，输出分数和报告。

---

## 📊 评测模式

### Code-Dev 模式（推荐快速评测）✅

```
┌──────────────┐
│ PaperBench   │
│   论文       │
└───────┬──────┘
        │
        ▼
┌──────────────────────────┐
│ MS-Agent Code Scratch    │
│ • 论文理解 & 分析         │
│ • 代码生成                │
│ • 质量检查                │
└───────┬──────────────────┘
        │
        ▼
     [评分]
   0-100 分
```

**特点**：
- ⚡ 快速（3 篇论文约 5-10 分钟）
- 💰 便宜（API 调用少）
- 📍 清晰（代码质量指标明确）

**运行**：
```bash
python evaluate_paperbench.py --split debug --type code-dev
```

### Complete 模式（完整严格评测）

包括代码生成、代码执行和结果验证。需要 GPU，成本较高，但评测更全面。

**运行**：
```bash
python evaluate_paperbench.py --split debug --type complete
```

---

## 📈 论文分割选项

| 模式 | 论文数 | 时间 | 成本 | 用途 |
|------|--------|------|------|------|
| `--split debug` | 3 篇 | 5-10 分钟 | 最低 | 测试和验证 |
| `--split mini` | 10 篇 | 30-60 分钟 | 中等 | 快速评估 |
| `--split full` | 20 篇 | 2-4 小时 | 高 | 完整评测 |

---

## 📋 推荐操作流程

### 新手（第一次使用）

```bash
# 1. 阅读快速开始
cat PAPERBENCH_QUICKSTART.md

# 2. 运行 debug 评测
python evaluate_paperbench.py --split debug --type code-dev

# 3. 查看结果
cat paperbench_results/*/results_final.json | python -m json.tool
```

### 进阶（优化性能）

```bash
# 1. 阅读完整评测方案
cat PAPERBENCH_EVALUATION.md

# 2. 调整 MS-Agent 提示词
# 编辑 architecture.yaml 和 refine.yaml

# 3. 运行 mini 评测
python evaluate_paperbench.py --split mini --type code-dev

# 4. 分析结果，识别薄弱点
# 查看低分论文的特征

# 5. 迭代改进
```

### 专家（全流程评测）

```bash
# 1. 研究 PaperBench 官方论文
# https://arxiv.org/abs/2504.01848

# 2. 完整评测（包括执行）
python evaluate_paperbench.py --split full --type complete

# 3. 生成详细报告
python -c "import json; data=json.load(open('paperbench_results/*/results_final.json')); ..."

# 4. 提交到官方排行榜（可选）
```

---

## 🎯 预期性能基准

根据 OpenAI 官方评测结果：

| 模型 | Code-Dev 分数 | 完整分数 | 备注 |
|------|-------------|--------|------|
| **Claude 3.5 Sonnet** | **21.0%** | 16.1% | 当前最好的开源模型 |
| o1-high (36h) | 26.0% | - | OpenAI 最新模型 |
| GPT-4o | 4.1% | - | 早期模型 |

**目标**：让 MS-Agent Code Scratch 在 Code-Dev 模式下 **超过 21.0%** 的 Claude 3.5 Sonnet 基线。

---

## 📊 输出和报告

### 结果存储位置

```
paperbench_results/
├── 20250116_120000/          # 运行时间戳
│   ├── results_temp.json     # 中间结果（运行中）
│   └── results_final.json    # 最终结果
├── 20250116_150000/
│   └── ...
```

### 结果格式示例

```json
{
  "metadata": {
    "timestamp": "2025-01-16T12:00:00",
    "split": "debug",
    "eval_type": "code-dev",
    "total_papers": 3
  },
  "summary": {
    "total": 3,
    "completed": 3,
    "failed": 0,
    "success_rate": 1.0,
    "average_score": 0.50
  },
  "papers": [
    {
      "paper_id": "dpo-direct-preference",
      "status": "completed",
      "score": 0.65,
      "code_generated": true,
      "compilation_passed": true
    },
    ...
  ]
}
```

---

## 🔧 常见问题

### Q: "PAPERBENCH_DATA_DIR 未设置"
```bash
# 检查设置
echo $PAPERBENCH_DATA_DIR  # Linux/Mac
echo $Env:PAPERBENCH_DATA_DIR  # Windows PowerShell

# 临时设置
export PAPERBENCH_DATA_DIR=/path/to/frontier-evals/project/paperbench/data

# 永久设置
# 编辑 ~/.bashrc (Linux/Mac) 或系统环境变量 (Windows)
```

### Q: "API Key 无效"
```bash
# 检查 API Key
echo $OPENAI_API_KEY | head -c 10

# 获取新的 Key
# https://platform.openai.com/api-keys

# 临时设置
export OPENAI_API_KEY=sk-xxx...
```

### Q: "找不到论文"
```bash
# 检查数据下载
ls $PAPERBENCH_DATA_DIR/papers/ | wc -l

# 应该输出 20（20 篇论文）

# 如果少于 20，重新拉取
cd frontier-evals
git lfs fetch --include "project/paperbench/data/**" --force
git lfs checkout project/paperbench/data --force
```

### Q: "评测太慢"
```bash
# 使用 debug 模式（仅 3 篇）
--split debug

# 使用 code-dev 模式（不需要 GPU）
--type code-dev
```

---

## 📚 进阶资源

### 官方文档
- **PaperBench 官网**：https://openai.com/index/paperbench/
- **论文**：https://arxiv.org/abs/2504.01848
- **GitHub**：https://github.com/openai/frontier-evals

### MS-Agent 文档
- **官方文档**：https://ms-agent.readthedocs.io/
- **GitHub**：https://github.com/modelscope/ms-agent

### 本项目文档
- 📖 **快速开始**：PAPERBENCH_QUICKSTART.md
- 📚 **完整方案**：PAPERBENCH_EVALUATION.md
- 📋 **项目总结**：PAPERBENCH_SUMMARY.md

---

## 💡 优化建议

### 短期（1-2 周）

1. **调整提示词**
   - 根据评分结果优化 system prompt
   - 重点关注低分论文

2. **扩展测试**
   - 从 debug 升级到 mini
   - 收集更多反馈

3. **性能目标**
   - 当前：需要确定
   - 目标：25-30%（超越 Claude 3.5）

### 中期（1-2 月）

1. **增强论文理解**
   - 支持 PDF 解析
   - 自动提取关键信息

2. **改进代码生成**
   - 更好的模块化设计
   - 自动测试用例生成

3. **优化编译修复**
   - 更智能的错误诊断
   - 上下文感知的修复

---

## 🤝 贡献指南

欢迎贡献！可以：
1. 提交改进建议（Issues）
2. 贡献优化代码（Pull Requests）
3. 分享评测结果和最佳实践
4. 改进文档和示例

---

## 📞 支持

- **问题排查**：查看各文档的常见问题部分
- **官方支持**：https://github.com/openai/frontier-evals/issues
- **社区讨论**：MS-Agent GitHub Discussions

---

## 📄 许可证

本评测框架遵循 MS-Agent 的原许可证。

PaperBench 官方资源受 OpenAI 许可管辖：
- https://github.com/openai/frontier-evals

---

## 🎉 快速开始命令

```bash
# 一行命令快速开始（假设环境已配置）
python projects/code_scratch/evaluate_paperbench.py --split debug --type code-dev

# 或使用启动脚本
bash projects/code_scratch/run_paperbench.sh debug code-dev
```

**祝你评测顺利！** 🚀

有任何问题，请查阅相关文档或提交 Issue。
