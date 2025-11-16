#!/usr/bin/env bash
# PaperBench 评测 - 快速操作清单
# 按照顺序执行这些命令即可开始 PaperBench 评测

set -e  # 任何错误立即退出

echo "======================================"
echo "MS-Agent PaperBench 评测快速启动"
echo "======================================"

# 1. 检查环境
echo -e "\n[1/5] 检查环境..."

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未安装 Python"
    exit 1
fi
echo "✓ Python 版本: $(python --version)"

# 检查 git lfs
if ! command -v git lfs &> /dev/null; then
    echo "⚠️  警告: 未安装 git lfs（需要下载 PaperBench 数据）"
    echo "   安装: https://git-lfs.com/"
fi

# 2. 准备 PaperBench 数据
echo -e "\n[2/5] 准备 PaperBench 数据..."

if [ -z "$PAPERBENCH_DATA_DIR" ]; then
    echo "⚠️  环境变量 PAPERBENCH_DATA_DIR 未设置"
    echo "   请执行: export PAPERBENCH_DATA_DIR=/path/to/frontier-evals/project/paperbench/data"
    echo ""
    echo "   完整步骤："
    echo "   1. git clone https://github.com/openai/frontier-evals.git --filter=blob:none"
    echo "   2. cd frontier-evals"
    echo "   3. git lfs fetch --include 'project/paperbench/data/**'"
    echo "   4. git lfs checkout project/paperbench/data"
    echo "   5. export PAPERBENCH_DATA_DIR=\$(pwd)/project/paperbench/data"
    exit 1
fi

if [ ! -d "$PAPERBENCH_DATA_DIR/papers" ]; then
    echo "❌ 错误: 找不到 $PAPERBENCH_DATA_DIR/papers"
    exit 1
fi

PAPER_COUNT=$(ls -d "$PAPERBENCH_DATA_DIR/papers"/*/ 2>/dev/null | wc -l)
echo "✓ 已找到 $PAPER_COUNT 篇论文"

# 3. 检查 API Key
echo -e "\n[3/5] 检查 API 密钥..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 环境变量 OPENAI_API_KEY 未设置"
    echo "   请执行: export OPENAI_API_KEY=sk-xxx..."
    exit 1
fi

echo "✓ OpenAI API Key 已设置"

# 4. 检查 MS-Agent 项目
echo -e "\n[4/5] 检查 MS-Agent 项目..."

if [ ! -d "projects/code_scratch" ]; then
    echo "❌ 错误: 找不到 projects/code_scratch"
    echo "   请确保在 MS-Agent 项目根目录运行本脚本"
    exit 1
fi

if [ ! -f "projects/code_scratch/evaluate_paperbench.py" ]; then
    echo "❌ 错误: 找不到 evaluate_paperbench.py"
    exit 1
fi

echo "✓ MS-Agent Code Scratch 项目已就绪"

# 5. 运行评测
echo -e "\n[5/5] 运行评测..."

EVAL_MODE="${1:-debug}"  # 默认 debug 模式
EVAL_TYPE="${2:-code-dev}"  # 默认 code-dev

echo ""
echo "评测配置:"
echo "  - 论文分割: $EVAL_MODE (debug=3篇, mini=10篇, full=20篇)"
echo "  - 评测类型: $EVAL_TYPE (code-dev=仅代码, complete=包含执行)"
echo ""

python projects/code_scratch/evaluate_paperbench.py \
    --split "$EVAL_MODE" \
    --type "$EVAL_TYPE" \
    --paperbench-dir "$PAPERBENCH_DATA_DIR"

echo ""
echo "======================================"
echo "✓ 评测完成！"
echo "======================================"
echo ""
echo "结果位置:"
ls -dt paperbench_results/*/ | head -1
echo ""
echo "查看详细结果:"
echo "  cat paperbench_results/*/results_final.json | python -m json.tool"
echo ""
