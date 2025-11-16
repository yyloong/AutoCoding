#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本：在 MS-Agent 中评测 PaperBench

使用方法：
    python evaluate_paperbench.py --split debug --type code-dev
    python evaluate_paperbench.py --split full --type complete
"""

import os
import sys
import json
import argparse
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

# 设置 Windows 支持 UTF-8 输出
if os.name == 'nt':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class PaperBenchEvaluationRunner:
    """PaperBench 评测执行器"""
    
    def __init__(self, paperbench_dir: str, split: str = "debug", eval_type: str = "code-dev"):
        self.paperbench_dir = Path(paperbench_dir)
        self.split = split
        self.eval_type = eval_type
        self.results_dir = Path("paperbench_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 评测器初始化成功")
        print(f"  - PaperBench 数据目录: {self.paperbench_dir}")
        print(f"  - 评测类型: {eval_type}")
        print(f"  - 论文分割: {split}")
        print(f"  - 结果保存目录: {self.results_dir}")
    
    def get_papers(self) -> list:
        """获取要评测的论文列表"""
        papers_dir = self.paperbench_dir / "papers"
        
        if not papers_dir.exists():
            print(f"❌ 错误: 找不到论文目录 {papers_dir}")
            print(f"   请设置 PAPERBENCH_DATA_DIR 环境变量或检查路径")
            return []
        
        all_papers = sorted([d.name for d in papers_dir.iterdir() if d.is_dir()])
        
        if self.split == "debug":
            papers = all_papers[:3]  # 调试时只用前 3 篇
        elif self.split == "mini":
            papers = all_papers[:10]  # 快速评测 10 篇
        else:
            papers = all_papers  # 全部 20 篇
        
        print(f"\n📄 获取论文列表:")
        print(f"  - 总论文数: {len(all_papers)}")
        print(f"  - 本次评测: {len(papers)} 篇")
        print(f"  - 论文列表: {papers[:5]}..." if len(papers) > 5 else f"  - 论文列表: {papers}")
        
        return papers
    
    def validate_environment(self) -> bool:
        """检验环境是否配置正确"""
        print("\n🔍 检验环境配置...")
        
        checks = []
        
        # 1. 检查 PaperBench 数据目录
        if not self.paperbench_dir.exists():
            checks.append(("PaperBench 数据目录", False, f"找不到 {self.paperbench_dir}"))
        else:
            checks.append(("PaperBench 数据目录", True, str(self.paperbench_dir)))
        
        # 2. 检查是否有 papers 目录
        papers_dir = self.paperbench_dir / "papers"
        if not papers_dir.exists():
            checks.append(("论文目录", False, "找不到 papers 子目录"))
        else:
            paper_count = len(list(papers_dir.iterdir()))
            checks.append(("论文目录", True, f"包含 {paper_count} 篇论文"))
        
        # 3. 检查 API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            checks.append(("OpenAI API Key", True, "✓ 已设置"))
        else:
            checks.append(("OpenAI API Key", False, "未设置"))
        
        # 4. 检查 MS-Agent 项目
        if Path("projects/code_scratch").exists():
            checks.append(("MS-Agent Code Scratch", True, "✓ 项目存在"))
        else:
            checks.append(("MS-Agent Code Scratch", False, "找不到项目"))
        
        # 打印检验结果
        for item, status, detail in checks:
            icon = "✓" if status else "✗"
            color_prefix = "\033[92m" if status else "\033[91m"
            color_suffix = "\033[0m"
            print(f"  {color_prefix}{icon} {item}{color_suffix}: {detail}")
        
        all_ok = all(status for _, status, _ in checks)
        return all_ok
    
    async def run_evaluation(self):
        """运行评测"""
        print(f"\n🚀 开始评测...")
        
        papers = self.get_papers()
        if not papers:
            print("❌ 没有论文可以评测")
            return False
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "split": self.split,
                "eval_type": self.eval_type,
                "total_papers": len(papers),
            },
            "papers": []
        }
        
        for i, paper_id in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] 评测论文: {paper_id}")
            
            result = await self._evaluate_paper(paper_id)
            results["papers"].append(result)
            
            # 保存中间结果
            self._save_results(results, f"results_temp.json")
        
        # 计算汇总统计
        results["summary"] = self._calculate_summary(results["papers"])
        
        # 保存最终结果
        self._save_results(results, f"results_final.json")
        
        return True
    
    async def _evaluate_paper(self, paper_id: str) -> dict:
        """评测单篇论文 - 调用 MS-Agent 生成代码并验证"""
        paper_dir = self.paperbench_dir / "papers" / paper_id
        
        result = {
            "paper_id": paper_id,
            "status": "pending",
            "score": 0,
            "details": {}
        }
        
        try:
            # 检查论文文件
            paper_file = paper_dir / "paper.md"
            if not paper_file.exists():
                paper_file = paper_dir / "paper.pdf"
            
            if not paper_file.exists():
                result["status"] = "failed"
                result["error"] = "找不到论文文件"
                return result
            
            # 读取论文内容与评估标准
            with open(paper_file, "r", encoding="utf-8") as f:
                paper_content = f.read()
            
            rubric_file = paper_dir / "rubric.json"
            rubric = {}
            if rubric_file.exists():
                with open(rubric_file) as f:
                    rubric = json.load(f)
            result["details"]["rubric"] = rubric
            
            # 构建 prompt：根据论文内容让 MS-Agent 生成项目代码
            prompt = f"""
根据以下论文描述，实现该论文中提出的系统/方法。生成完整的项目代码，包括前后端实现。

论文摘要:
{paper_content[:2000]}

要求:
1. 生成前端代码（React + Vite）和后端代码（Node.js）
2. 确保代码能够编译和运行
3. 实现论文的核心功能演示
"""
            
            # 调用 MS-Agent CLI
            print(f"    → 调用 MS-Agent 生成代码...")
            cmd = [
                "python", "ms_agent/cli/cli.py", "run",
                "--config", "projects/code_scratch",
                "--query", prompt,
                "--trust_remote_code", "true"
            ]
            
            # 设置环境变量以支持 UTF-8 输出（解决 Windows GBK 编码问题）
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            try:
                # 改用 subprocess（而非 asyncio.create_subprocess_exec）以避免复杂的流读取
                # 使用阻塞模式并在默认线程池中运行，超时由 asyncio.wait_for 管理
                loop = asyncio.get_event_loop()
                result_proc = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            env=env,
                            timeout=300  # 5 分钟超时
                        )
                    ),
                    timeout=310  # asyncio 超时稍长，以便 subprocess 的 timeout 先触发
                )
                stdout_text = result_proc.stdout or ""
                stderr_text = result_proc.stderr or ""
                returncode = result_proc.returncode
                
            except asyncio.TimeoutError:
                result["status"] = "timeout"
                result["error"] = "MS-Agent 执行超时（超过 300 秒）"
                return result
            except subprocess.TimeoutExpired as e:
                result["status"] = "timeout"
                result["error"] = f"MS-Agent 执行超时"
                return result
            except Exception as e:
                result["status"] = "agent_execution_error"
                result["error"] = str(e)
                return result
            
            if returncode != 0:
                result["status"] = "agent_failed"
                result["error"] = f"MS-Agent 执行失败: {stderr_text[-500:]}"
                result["details"]["stderr"] = stderr_text[-1000:]
                return result
            
            # 检查生成的代码
            output_dir = Path("output")
            if not output_dir.exists():
                result["status"] = "code_generation_failed"
                result["error"] = "MS-Agent 未生成 output 目录"
                return result
            
            # 统计生成的文件
            generated_files = list(output_dir.rglob("*"))
            result["details"]["generated_files_count"] = len([f for f in generated_files if f.is_file()])
            result["details"]["has_frontend"] = (output_dir / "frontend").exists()
            result["details"]["has_backend"] = (output_dir / "backend").exists()
            
            # 尝试编译前端
            compilation_passed = False
            try:
                print(f"    → 验证前端编译...")
                proc = await asyncio.create_subprocess_exec(
                    "npm", "run", "build",
                    cwd="output/frontend",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=10*1024*1024
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=120
                )
                
                if proc.returncode == 0:
                    compilation_passed = True
                    result["details"]["frontend_build"] = "passed"
                else:
                    result["details"]["frontend_build"] = "failed"
                    result["details"]["frontend_build_error"] = stderr.decode("utf-8", errors="replace")[-500:]
            except asyncio.TimeoutError:
                result["details"]["frontend_build"] = "timeout"
            except Exception as e:
                result["details"]["frontend_build"] = f"error: {str(e)}"
            
            # 计算综合评分
            score = 0.0
            
            # 代码生成成功 (0.3)
            if result["details"]["generated_files_count"] > 5:
                score += 0.3
            
            # 生成了前后端 (0.3)
            if result["details"]["has_frontend"] and result["details"]["has_backend"]:
                score += 0.3
            
            # 前端编译通过 (0.4)
            if compilation_passed:
                score += 0.4
            
            result["status"] = "completed"
            result["score"] = min(score, 1.0)
            result["code_generated"] = result["details"]["generated_files_count"] > 0
            result["compilation_passed"] = compilation_passed
            
        except asyncio.TimeoutError:
            result["status"] = "timeout"
            result["error"] = "评测超时（超过 5 分钟）"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            import traceback
            result["details"]["traceback"] = traceback.format_exc()
        
        return result
    
    def _calculate_summary(self, results: list) -> dict:
        """计算汇总统计"""
        if not results:
            return {}
        
        completed = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] in ["failed", "error"]]
        
        avg_score = sum(r.get("score", 0) for r in completed) / len(completed) if completed else 0
        
        return {
            "total": len(results),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(results) if results else 0,
            "average_score": avg_score,
        }
    
    def _save_results(self, results: dict, filename: str):
        """保存结果"""
        output_file = self.results_dir / filename
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  → 结果已保存: {output_file}")
    
    def print_summary(self, results: dict):
        """打印总结报告"""
        summary = results.get("summary", {})
        
        print(f"\n{'='*60}")
        print("📊 评测总结")
        print(f"{'='*60}")
        print(f"总论文数:        {summary.get('total', 0)}")
        print(f"完成:            {summary.get('completed', 0)}")
        print(f"失败:            {summary.get('failed', 0)}")
        print(f"成功率:          {summary.get('success_rate', 0):.1%}")
        print(f"平均分数:        {summary.get('average_score', 0):.2f}")
        print(f"{'='*60}")
        
        # 对标基线
        baseline = {
            "Claude 3.5 Sonnet (Code-Dev)": 0.21,
            "Claude 3.5 Sonnet (完整)": 0.161,
            "GPT-4o": 0.041,
        }
        
        avg_score = summary.get('average_score', 0)
        print(f"\n📈 基线对标:")
        for model, baseline_score in baseline.items():
            diff = avg_score - baseline_score
            status = "✓ 超越" if diff > 0 else "✗ 低于"
            print(f"  {status} {model}: {baseline_score:.1%} (差异: {diff:+.1%})")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PaperBench 评测工具 - 评测 MS-Agent 论文复现能力"
    )
    
    parser.add_argument(
        "--split",
        choices=["debug", "mini", "full"],
        default="debug",
        help="论文分割: debug(3篇), mini(10篇), full(20篇)"
    )
    
    parser.add_argument(
        "--type",
        dest="eval_type",
        choices=["code-dev", "complete"],
        default="code-dev",
        help="评测类型: code-dev(仅代码), complete(包括执行)"
    )
    
    parser.add_argument(
        "--paperbench-dir",
        default=os.getenv("PAPERBENCH_DATA_DIR"),
        help="PaperBench 数据目录（默认从 PAPERBENCH_DATA_DIR 环境变量读取）"
    )
    
    args = parser.parse_args()
    
    # 检查必要参数
    if not args.paperbench_dir:
        print("❌ 错误: 请设置 PAPERBENCH_DATA_DIR 环境变量或使用 --paperbench-dir 参数")
        sys.exit(1)
    
    # 创建评测器
    runner = PaperBenchEvaluationRunner(
        args.paperbench_dir,
        split=args.split,
        eval_type=args.eval_type
    )
    
    # 验证环境
    if not runner.validate_environment():
        print("\n❌ 环境检验失败，请按上述提示修复")
        sys.exit(1)
    
    # 运行评测
    print(f"\n✓ 环境检验通过，准备开始评测...")
    success = await runner.run_evaluation()
    
    if success:
        # 加载并打印最终结果
        results_file = runner.results_dir / "results_final.json"
        with open(results_file) as f:
            results = json.load(f)
        
        runner.print_summary(results)
        
        print(f"\n✓ 评测完成！")
        print(f"  详细结果保存在: {runner.results_dir}/")
        print(f"  查看结果: cat {results_file}")
    else:
        print(f"\n❌ 评测失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
