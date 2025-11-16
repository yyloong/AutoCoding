# MS-Agent Code Scratch - PaperBench 评测指南

## 📋 概述

本文档说明如何使用 **PaperBench** 基准来评测 **MS-Agent Code Scratch** 项目在 AI 论文复现中的能力。

PaperBench 是 OpenAI 开源的评测基准，包含 20 篇 ICML 2024 论文，共 8,316 个可评分任务。

**MS-Agent Code Scratch** 是一个代码生成和修复系统，特别适合在 PaperBench 上进行评测，因为它具备：
- ✅ 论文内容理解与分析
- ✅ 代码项目自动生成
- ✅ 编译错误检测与修复
- ✅ 完整的 workflow 支持

---

## 🎯 评测方案设计

### 方案 1: Code-Dev 评测（推荐快速评测）

**适用场景**：快速评估代码生成质量，无需 GPU 和实验执行

**特点**：
- 仅评估代码开发质量（不评估执行和结果匹配）
- 无需 GPU，成本和时间降低 ~85%
- 评估指标清晰，可快速迭代

**流程**：
```
PaperBench Paper 
    ↓
MS-Agent 论文分析 → 代码生成 → 代码质量评分
    ↑
评估：方法实现、数据处理、模块结构
```

### 方案 2: 完整评测（更严格）

**适用场景**：完整评估代码复现能力，包括实验执行

**特点**：
- 评估代码开发 + 代码执行 + 结果匹配
- 需要 GPU 和实验环境
- 更接近真实研究复现

**流程**：
```
PaperBench Paper 
    ↓
MS-Agent 论文分析 → 代码生成 → 代码执行 → 结果验证 → 综合评分
    ↑
评估：代码质量 + 执行成功 + 结果准确性
```

---

## 📦 安装和准备

### 1. 安装 PaperBench

```bash
# 克隆官方仓库
git clone https://github.com/openai/frontier-evals.git --filter=blob:none
cd frontier-evals

# 下载数据集（使用 Git LFS）
git lfs fetch --include "project/paperbench/data/**"
git lfs checkout project/paperbench/data

# 设置环境变量
export PAPERBENCH_DATA_DIR="$(pwd)/project/paperbench/data"
```

### 2. 安装依赖

```bash
# 进入 PaperBench 目录
cd project/paperbench

# 使用 uv 安装依赖
uv sync
```

### 3. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件，填入必要的 API Key
# - OPENAI_API_KEY: 用于 Agent
# - GRADER_OPENAI_API_KEY: 用于评分（可选，默认同上）
# - 其他模型 API Key（如需要）
```

### 4. 准备 Agent 资源（如需要）

```bash
# 某些论文需要额外权限
cp paperbench/agents/agent.env.example paperbench/agents/agent.env

# 编辑 agent.env，填入：
# - OPENAI_API_KEY（用于 API 调用）
# - HF_TOKEN（HuggingFace token，用于 ImageNet/Llama-2）
```

### 5. 构建 Docker 镜像

```bash
# 构建所有必要的 Docker 镜像
bash paperbench/scripts/build-docker-images.sh

# 或手动构建基础镜像
docker build -f paperbench/Dockerfile.base -t pb-env:latest .
docker build -f paperbench/reproducer.Dockerfile -t pb-reproducer:latest .
```

---

## 🚀 评测执行步骤

### 步骤 1: 修改 MS-Agent 配置以支持 PaperBench

在 `projects/code_scratch/refine.yaml` 中添加或修改：

```yaml
paperbench:
  enabled: true
  eval_type: "code-dev"  # 或 "full"
  paper_dir: "${PAPERBENCH_DATA_DIR}"
  
prompt:
  system: |
    你是一名高级研究工程师，你的任务是复现学术论文。

    流程：
    1. 仔细阅读论文（PDF 或 Markdown）
    2. 理解论文的核心贡献和方法
    3. 设计完整的代码架构
    4. 实现所有关键方法
    5. 准备测试代码和数据处理
    
    请按照以下格式输出：
    - 论文理解总结
    - 代码结构设计
    - 关键实现细节
    - 测试计划
```

### 步骤 2: 创建 PaperBench 评测包装器

创建文件 `projects/code_scratch/paperbench_wrapper.py`：

```python
#!/usr/bin/env python3
"""
PaperBench 评测包装器，集成 MS-Agent Code Scratch
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

class PaperBenchEvaluator:
    """使用 MS-Agent 评测 PaperBench 论文"""
    
    def __init__(self, paperbench_data_dir: str, ms_agent_config: str):
        self.paperbench_dir = Path(paperbench_data_dir)
        self.ms_agent_config = ms_agent_config
        self.results = []
    
    async def evaluate_paper(self, paper_id: str) -> Dict[str, Any]:
        """评测单篇论文"""
        paper_dir = self.paperbench_dir / "papers" / paper_id
        
        # 读取论文信息
        paper_md = paper_dir / "paper.md"
        if not paper_md.exists():
            paper_md = paper_dir / "paper.pdf"  # 备选
        
        # 读取评估标准（Rubric）
        rubric_file = paper_dir / "rubric.json"
        with open(rubric_file) as f:
            rubric = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"正在评测论文: {paper_id}")
        print(f"{'='*60}")
        
        # 调用 MS-Agent Code Scratch
        result = await self._run_ms_agent(paper_id, paper_dir)
        
        return {
            "paper_id": paper_id,
            "status": result.get("status", "failed"),
            "score": result.get("score", 0),
            "code_generated": result.get("code_generated", False),
            "compilation_passed": result.get("compilation_passed", False),
            "errors": result.get("errors", []),
            "rubric": rubric
        }
    
    async def _run_ms_agent(self, paper_id: str, paper_dir: Path) -> Dict:
        """运行 MS-Agent 代码生成"""
        # 这里调用 MS-Agent CLI
        query = f"""
        复现以下论文：{paper_id}
        
        论文目录：{paper_dir}
        
        请：
        1. 阅读论文内容
        2. 理解论文的方法和贡献
        3. 生成完整的代码实现
        4. 确保代码能成功编译
        """
        
        # 调用 MS-Agent 的 LLM Agent
        # 这里需要集成实际的 agent 调用逻辑
        
        return {
            "status": "completed",
            "code_generated": True,
            "compilation_passed": True,
            "score": 0.5
        }
    
    async def evaluate_all(self, paper_split: str = "debug") -> List[Dict]:
        """评测所有论文"""
        papers_dir = self.paperbench_dir / "papers"
        
        # 根据 split 过滤论文
        if paper_split == "debug":
            # 调试用，只评测少数几篇
            paper_ids = [d.name for d in papers_dir.iterdir() 
                        if d.is_dir()][:3]
        else:
            # 完整评测
            paper_ids = [d.name for d in papers_dir.iterdir() 
                        if d.is_dir()]
        
        for paper_id in paper_ids:
            result = await self.evaluate_paper(paper_id)
            self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成评测报告"""
        if not self.results:
            return {}
        
        total_papers = len(self.results)
        successful = sum(1 for r in self.results 
                        if r["status"] == "completed")
        avg_score = sum(r.get("score", 0) for r in self.results) / total_papers
        compilation_rate = sum(1 for r in self.results 
                              if r.get("compilation_passed")) / total_papers
        
        return {
            "total_papers": total_papers,
            "successful_completions": successful,
            "success_rate": successful / total_papers,
            "average_score": avg_score,
            "compilation_pass_rate": compilation_rate,
            "detailed_results": self.results
        }


async def main():
    """主评测流程"""
    paperbench_dir = os.getenv("PAPERBENCH_DATA_DIR")
    if not paperbench_dir:
        raise ValueError("请设置 PAPERBENCH_DATA_DIR 环境变量")
    
    ms_agent_config = "projects/code_scratch"
    
    evaluator = PaperBenchEvaluator(paperbench_dir, ms_agent_config)
    
    # 运行评测（先用 debug 分割测试）
    await evaluator.evaluate_all("debug")
    
    # 生成报告
    report = evaluator.generate_report()
    
    print("\n" + "="*60)
    print("评测报告")
    print("="*60)
    print(f"总论文数: {report['total_papers']}")
    print(f"成功完成: {report['successful_completions']}")
    print(f"成功率: {report['success_rate']:.2%}")
    print(f"平均分数: {report['average_score']:.2f}")
    print(f"编译通过率: {report['compilation_pass_rate']:.2%}")
    
    # 保存报告
    with open("paperbench_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n报告已保存到 paperbench_report.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### 步骤 3: 运行 PaperBench Code-Dev 评测

```bash
# 快速评测（仅代码质量）
cd frontier-evals/project/paperbench

uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=debug \
  paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
  paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
  paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
  paperbench.judge.code_only=True \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

### 步骤 4: 运行完整评测（可选）

```bash
# 完整评测（包括代码执行和结果验证）
# 需要 GPU 支持
uv run python -m paperbench.nano.entrypoint \
  paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
  paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
  paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
  paperbench.solver.is_nvidia_gpu_env=True \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

---

## 📊 评估指标

### Code-Dev 评估指标

| 指标 | 说明 | 权重 |
|------|------|------|
| **论文理解** | 是否正确理解论文主要贡献 | 20% |
| **方法实现** | 关键算法和方法的实现完整性 | 40% |
| **代码质量** | 代码结构、可读性、健壮性 | 20% |
| **数据处理** | 正确处理数据输入和输出 | 20% |

### 完整评测额外指标

| 指标 | 说明 | 权重 |
|------|------|------|
| **编译/执行** | 代码能否成功编译和运行 | 30% |
| **结果匹配** | 复现结果是否与论文结果一致 | 30% |

### 整体评分

```
总分 = 0-100 分

优秀 (80-100): 代码质量高，逻辑清晰，可直接使用
良好 (60-79): 代码可用，有少量问题
一般 (40-59): 代码有明显缺陷，需要修复
不合格 (0-39): 代码不可用或有严重问题
```

---

## 📈 结果分析

### 生成报告示例

```json
{
  "evaluation_summary": {
    "total_papers": 20,
    "papers_completed": 18,
    "papers_failed": 2,
    "average_score": 42.5,
    "success_rate": 0.90,
    "code_generation_rate": 0.95,
    "compilation_pass_rate": 0.85
  },
  "detailed_results": [
    {
      "paper_id": "dpo-direct-preference",
      "status": "completed",
      "code_generated": true,
      "compilation_passed": true,
      "score": 65.0,
      "execution_success": true,
      "result_match_score": 0.92,
      "time_spent_minutes": 45,
      "issues": []
    },
    ...
  ]
}
```

### 分析关键

```python
# 识别薄弱环节
- 哪些类型的论文评分低？（e.g., 硬件密集型、数据处理复杂）
- 代码生成还是编译修复环节有问题？
- 是否某类论文特别难？

# 对标基线
- Claude 3.5 Sonnet Code-Dev: 21.0%
- Claude 3.5 Sonnet (完整): 16.1%
- GPT-4o: 4.1%
```

---

## 🔧 集成建议

### 扩展 MS-Agent 支持

修改 `projects/code_scratch/config_handler.py`：

```python
class ConfigHandler(ConfigLifecycleHandler):
    def task_begin(self, config: DictConfig, tag: str) -> DictConfig:
        # 检测是否为 PaperBench 任务
        if hasattr(config, 'paperbench') and config.paperbench.enabled:
            # 切换到 PaperBench 特定配置
            config.callbacks = [
                'callbacks/paperbench_callback',
                'callbacks/artifact_callback'
            ]
            config.tools = {
                'paper_analyzer': {'type': 'pdf_reader'},
                'file_system': {'mcp': False},
                ...
            }
        return config
```

### 创建 PaperBench 回调

创建 `projects/code_scratch/callbacks/paperbench_callback.py`：

```python
class PaperBenchCallback(Callback):
    """为 PaperBench 评测优化的回调"""
    
    async def on_task_begin(self, runtime, messages):
        # 加载论文和评估标准
        self.paper_info = self._load_paper_info()
        self.rubric = self._load_rubric()
    
    async def on_generate_response(self, runtime, messages):
        # 检验代码生成质量
        self._validate_code_against_rubric()
    
    def _load_paper_info(self):
        # 从 PAPERBENCH_DATA_DIR 读取论文
        ...
    
    def _load_rubric(self):
        # 读取评估标准 JSON
        ...
    
    def _validate_code_against_rubric(self):
        # 根据标准检验代码
        ...
```

---

## 💡 最佳实践

1. **从 Debug 开始**：先用 `paper_split=debug` 测试 3-5 篇论文
2. **逐步扩展**：确认流程无误后再全量评测
3. **监测成本**：注意 API 调用成本，Code-Dev 成本低
4. **保存日志**：详细日志用于事后分析
5. **版本跟踪**：记录每次评测的 MS-Agent 版本和配置

---

## 🎓 参考资源

- **PaperBench 官网**：https://openai.com/index/paperbench/
- **GitHub 代码**：https://github.com/openai/frontier-evals/tree/main/project/paperbench
- **论文**：https://arxiv.org/abs/2504.01848
- **MS-Agent 文档**：https://ms-agent.readthedocs.io/

---

## 📞 常见问题

**Q: Code-Dev 和完整评测哪个更合适？**
A: 快速迭代用 Code-Dev（成本低），最终评估用完整评测（更严格）。

**Q: 需要什么样的 GPU？**
A: Code-Dev 不需要 GPU，完整评测建议 A100/H100。

**Q: 评分多少才算成功？**
A: 当前基线：Claude 3.5 Sonnet Code-Dev 约 21%，超过这个分数即为优于基线。

**Q: 如何调试失败的论文？**
A: 查看 `runs/` 目录下的详细日志，使用 `uv run python paperbench/gui/app.py` 查看评估标准。
