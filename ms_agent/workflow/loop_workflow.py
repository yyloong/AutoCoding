import os
import shutil
from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from ms_agent.llm import Message
from omegaconf import DictConfig
from ms_agent.tools.file_parser_tools.file_parser import SingleFileParser

logger = get_logger()

class LoopWorkflow(Workflow):
    """A workflow implementation with loop support between coder and refiner agents."""

    WORKFLOW_NAME = 'LoopWorkflow'

    def build_workflow(self):
        """Build the workflow chain from config."""
        print(self.config)
        if not self.config:
            return
        reserved_keys = ['type', 'local_dir', 'name', 'tools', 'callbacks']
        self.workflow_chains = [key for key in self.config.keys() if key not in reserved_keys]
        logger.info(f'Workflow chains: {self.workflow_chains}')

    async def run(self, inputs, **kwargs):
        """
        Execute the workflow with loop support between coder and refiner.
        
        Workflow:
        1. researcher: 调研并生成 analysis.md
        2. planner: 生成开发计划 plan.md
        3. loop:
           - coder: 根据 plan.md 和 report.md（如果有）编写代码，生成 README.md
           - refiner: 运行代码并审查，生成 report.md（包含 accept 字段）
           - 如果 accept=true，退出循环；否则继续循环
        """
        # clean 当前目录下的 output 和 memory 目录
        if os.path.exists('output'):
            shutil.rmtree('output')
        #if os.path.exists('memory'):
        #    shutil.rmtree('memory')
        if self.input_file_path: 
            if not os.path.exists(self.input_file_path):
                raise FileNotFoundError(f"Input file not found: {self.input_file_path}")
            parser = SingleFileParser(cfg={
                'path': os.getcwd()
            })
            input_params = {"url": self.input_file_path}
            file_parser_result = parser.call(input_params)
            inputs = inputs + "\n\ninput file content:\n" + file_parser_result


        max_iterations = kwargs.get('max_iterations', 10)
        iteration_count = 0
        idx = 0

        for task in self.workflow_chains:
            if task != 'loop':
                # 执行非循环任务（researcher, planner）
                task_info = self.config[task]
                config = task_info['agent_config']
                
                if not hasattr(task_info, 'agent'):
                    task_info.agent = DictConfig({})
                
                init_args = getattr(task_info.agent, 'kwargs', {})
                init_args.pop('trust_remote_code', None)
                init_args['trust_remote_code'] = self.trust_remote_code
                init_args['mcp_server_file'] = self.mcp_server_file
                init_args['task'] = task
                init_args['load_cache'] = self.load_cache
                if isinstance(config, str):
                    init_args['config_dir_or_id'] = os.path.join(self.config.local_dir, config)
                else:
                    init_args['config'] = config
                init_args['env'] = self.env
                if 'tag' not in init_args:
                    init_args['tag'] = task
                
                logger.info(f'Running task: {task}')
                agent = AgentLoader.build(**init_args)

                outputs = await agent.run(inputs)
                logger.info(f'Task {task} completed')

                idx += 1

            else:
                # 执行循环任务（coder <-> refiner）
                tasks = self.config[task]
                import itertools
                
                for task_name in itertools.cycle(tasks.keys()):
                    # 检查是否超过最大迭代次数
                    if iteration_count >= max_iterations:
                        logger.warning(f'Reached maximum iterations ({max_iterations}), exiting loop.')
                        break
                    
                    task_info = tasks[task_name]
                    config = task_info['agent_config']
                    
                    if not hasattr(task_info, 'agent'):
                        task_info.agent = DictConfig({})
                    
                    init_args = getattr(task_info.agent, 'kwargs', {})
                    init_args.pop('trust_remote_code', None)
                    init_args['trust_remote_code'] = self.trust_remote_code
                    init_args['mcp_server_file'] = self.mcp_server_file
                    init_args['task'] = task_name
                    init_args['load_cache'] = self.load_cache
                    if isinstance(config, str):
                        init_args['config_dir_or_id'] = os.path.join(self.config.local_dir, config)
                    else:
                        init_args['config'] = config
                    init_args['env'] = self.env
                    if 'tag' not in init_args:
                        init_args['tag'] = task_name
                    
                    # 如果是 coder 任务，注入 plan.md 和 report.md 的内容
                    if task_name == 'coder':
                        inputs = self._inject_plan_and_feedback(inputs)
                        iteration_count += 1
                        logger.info(f'Running task: {task_name} (iteration: {iteration_count}/{max_iterations})')
                    else:
                        logger.info(f'Running task: {task_name}')
                    
                    agent = AgentLoader.build(**init_args)
                    outputs = await agent.run(inputs)
                    logger.info(f'Task {task_name} completed')
                    
                    # 如果是 breaker 任务（refiner），检查是否应该退出循环
                    if task_info.get('breaker', False):
                        if self._should_break_loop(outputs):
                            logger.info('Code accepted by refiner, exiting loop.')
                            break
                        else:
                            logger.info('Code not accepted, continuing to next iteration.')
                            # 清理 report.md 以便下次循环使用新的反馈
                            # （实际上 refiner 会重新生成 report.md）
        return inputs

    def _inject_plan_and_feedback(self, inputs):
        """
        在 coder 运行前，根据是否存在 report.md 来决定注入的内容和 system prompt。
        
        首次运行（report.md 不存在）：
        - 注入首次开发的 system prompt
        - 注入 plan.md 的内容
        
        迭代修改（report.md 存在）：
        - 注入迭代修改的 system prompt
        - 注入 README.md 和 report.md 的内容
        
        Args:
            inputs: 用户的原始输入（字符串或消息列表）
        Returns:
            增强后的输入
        """
        output_dir = self.config.get('output_dir', 'output')
        plan_path = os.path.join(output_dir, 'plan.md')
        report_path = os.path.join(output_dir, 'report.md')
        # 递归查找 README.md，可能在 output_dir 的子目录下
        readme_path = None
        for root, dirs, files in os.walk(output_dir):
            if 'README.md' in files:
                readme_path = os.path.join(root, 'README.md')
                break

        additional_context = "\n\n" + "="*80 + "\n"
        
        if os.path.exists(report_path):
            # 迭代修改模式
            logger.info("Coder mode: ITERATION (修改现有代码)")
            additional_context += self._get_iteration_system_prompt()
            additional_context += "\n" + "="*80 + "\n\n"
            additional_context += "## 现有项目信息和反馈修改意见\n\n"
            
            # 注入 README.md
            if readme_path is not None and os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    additional_context += "### 项目说明 (README.md):\n\n"
                    additional_context += readme_content + "\n\n"
                    logger.info(f"Injected README.md into coder input")
                except Exception as e:
                    logger.warning(f"Failed to read README.md: {e}")
            else:
                logger.warning("No README.md found to inject")
            
            # 注入 report.md
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                additional_context += "### 上一轮审查反馈 (report.md):\n\n"
                additional_context += report_content + "\n\n"
                additional_context += "**重要提醒**: 请优先修复以上审查反馈中的问题，按优先级处理（致命错误 → 功能缺陷 → 质量问题 → 文档问题）。\n\n"
                logger.info(f"Injected report.md into coder input")
            except Exception as e:
                logger.warning(f"Failed to read report.md: {e}")
        else:
            # 首次开发模式
            logger.info("Coder mode: INITIAL (首次开发)")
            additional_context += self._get_initial_system_prompt()
            additional_context += "\n" + "="*80 + "\n\n"
            additional_context += "## 开发计划\n\n"
            
            # 注入 plan.md
            if os.path.exists(plan_path):
                try:
                    with open(plan_path, 'r', encoding='utf-8') as f:
                        plan_content = f.read()
                    additional_context += "### 开发计划 (plan.md):\n\n"
                    additional_context += plan_content + "\n\n"
                    logger.info(f"Injected plan.md into coder input")
                except Exception as e:
                    logger.warning(f"Failed to read plan.md: {e}")
            else:
                logger.warning(f"plan.md not found at {plan_path}")

        additional_context += "="*80 + "\n"
        
        # 将额外的上下文添加到输入中
        if isinstance(inputs, str):
            return inputs + additional_context
        elif isinstance(inputs, list):
            # 假设 inputs 是消息列表，添加一个系统消息
            enhanced_inputs = inputs.copy()
            enhanced_inputs.append(Message(role='system', content=additional_context))
            return enhanced_inputs
        else:
            logger.warning(f"Unknown input type: {type(inputs)}, returning as is")
            return inputs
    
    def _get_initial_system_prompt(self):
        """返回首次开发模式的 system prompt"""
        return """
## 首次开发模式

你现在处于**首次开发模式**，需要从零开始实现项目。

### 工作流程

**步骤 1：理解开发计划**
- 仔细阅读下面提供的 `plan.md` 内容
- 理解项目的整体架构和开发步骤
- 如需了解技术细节，可使用 file_system 工具读取 `analysis.md`

**步骤 2：分步实现功能**
按照 plan.md 中的步骤顺序，逐步实现每个功能：

对于每一步：
a) 明确当前步骤的目标和任务清单
b) 设计代码结构（如需要多个函数/类，先规划好）
c) 编写代码实现：
   - 使用 file_system 工具创建文件到 output 目录下（**不要创建子目录**）
   - 代码要清晰、有适当注释
   - 遵循语言最佳实践（PEP 8 for Python 等）
   - 包含必要的错误处理
d) 验证代码：
   - 使用 run_code 工具运行代码文件（参数：file="文件名", language="python"）
   - 检查是否有语法错误或运行时错误
   - 如果有错误，立即修复并重新验证
   - 确保代码能正常运行后再进入下一步
e) 在输出中标注当前步骤已完成

**步骤 3：必要时进行调研**
如果遇到不确定的技术问题：
- 使用 web_search 工具搜索相关技术文档
- 查看 API 文档、示例代码等
- 基于搜索结果调整实现方案

**步骤 4：生成项目文档**
所有功能实现完成后，创建 `README.md` 文件到 output 目录下，包含：
- 项目简介和功能特性
- 技术栈
- 安装步骤（环境要求、依赖安装）
- 使用方法（基本使用、配置说明、使用示例）
- 项目结构说明
- 注意事项和常见问题

**步骤 5：最终验证**
- 按照 README.md 中的说明，完整运行一遍项目
- 确保所有功能都能正常工作
- 确认 README.md 的说明准确无误

**步骤 6：退出任务**
确认所有步骤完成、README.md 已创建后，使用 exit_task 工具退出任务。

### 代码质量要求
- 代码结构清晰，模块化设计
- 函数和变量命名有意义
- 适当的代码注释（不要过度注释）
- 包含必要的错误处理和输入验证
- 遵循语言规范

### 重要提醒
- 严格按照 plan.md 的步骤顺序实现
- 每一步完成后都要验证代码能运行
- 所有文件创建到 output 目录下，不要创建子目录
- 代码要实用、可运行，不要写伪代码或框架代码
- README.md 必须详细、准确，用户能按照说明成功运行项目
- 完成所有任务后才能退出

现在开始首次开发工作。
"""
    
    def _get_iteration_system_prompt(self):
        """返回迭代修改模式的 system prompt"""
        return """
## 迭代修改模式

你现在处于**迭代修改模式**，需要根据审查反馈修改现有代码。

### 工作流程

**步骤 1：理解现有项目和反馈**
- 仔细阅读下面提供的 `README.md` 内容，了解项目的当前状态
- 仔细阅读下面提供的 `report.md` 内容，重点关注：
  * **accept** 字段：为 false 说明需要修改
  * **问题清单**：列出了所有需要修复的问题（致命错误、功能缺陷、质量问题、文档问题）
  * **改进建议**：具体的修复方法和优化建议
- 使用 file_system 工具读取现有的代码文件，了解当前实现

**步骤 2：制定修复计划**
根据 report.md 中的问题清单，按优先级排序：
1. **致命错误**：必须立即修复（代码无法运行、依赖缺失、语法错误等）
2. **功能缺陷**：核心功能问题（功能不完整、逻辑错误、输出不正确等）
3. **质量问题**：代码质量改进（结构混乱、缺少注释、不符合规范等）
4. **文档问题**：README.md 的完善（说明不清、示例错误等）

**步骤 3：逐项修复问题**
对于每个问题：
a) 定位问题所在的文件和代码位置
b) 分析问题的根本原因
c) 设计修复方案（**最小化修改**，避免引入新问题）
d) 使用 file_system 工具修改代码文件
e) 验证修复：
   - 使用 run_code 工具运行代码（参数：file="文件名", language="python"）
   - 确认问题已解决
   - 确保没有引入新的错误
f) 在输出中标注该问题已修复

**步骤 4：完善未完成的功能**
如果 report.md 指出有功能未完成：
- 参考 report.md 中的建议
- 补充实现缺失的功能
- 确保新功能与现有代码良好集成

**步骤 5：更新项目文档**
如果修改了功能或修复了重要问题：
- 使用 file_system 工具更新 `README.md`
- 更新使用方法、注意事项等相关部分
- 如果有新增功能，添加到功能特性列表

**步骤 6：全面测试**
- 按照 README.md 的说明，完整运行项目
- 测试所有修复的功能
- 确保项目能正常运行

**步骤 7：退出任务**
确认所有问题已修复、代码能正常运行后，使用 exit_task 工具退出任务。

### 修改原则
- **针对性修复**：只修复 report.md 中指出的问题
- **最小化修改**：避免不必要的改动，减少引入新问题的风险
- **立即验证**：每次修改后立即测试，确保问题已解决
- **同步文档**：如果修改了功能，同步更新 README.md

### 代码质量要求
- 代码结构清晰，模块化设计
- 函数和变量命名有意义
- 适当的代码注释
- 包含必要的错误处理和输入验证
- 遵循语言规范

### 重要提醒
- 优先修复 report.md 中的问题，按优先级处理
- 每修复一个问题都要验证代码能运行
- 最小化修改，避免破坏正常功能
- 所有文件都在 output 目录下，不要创建子目录
- 如果修改了功能，同步更新 README.md
- 完成所有修复后才能退出

现在开始迭代修改工作。
"""

    def _should_break_loop(self, outputs):
        """
        判断是否应该退出循环，通过检查 refiner 的 report.md 中的 accept 字段。
        
        Args:
            outputs: refiner agent 的输出消息列表
            
        Returns:
            bool: True 表示应该退出循环（accept=true），False 表示继续循环
        """        
        output_dir = self.config.get('output_dir', 'output')
        report_path = os.path.join(output_dir, 'report.md')
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                # 查找 accept 字段
                import re
                match = re.search(r'\*\*accept\*\*:\s*(true|false)', report_content, re.IGNORECASE)
                if not match:
                    match = re.search(r'accept:\s*(true|false)', report_content, re.IGNORECASE)
                
                if match:
                    accept_value = match.group(1).lower()
                    logger.info(f"Found accept field in report.md: {accept_value}")
                    return accept_value == 'true'
                else:
                    logger.warning("Could not find accept field in report.md")
            except Exception as e:
                logger.warning(f"Failed to read report.md: {e}")
        else:
            logger.warning(f"report.md not found at {report_path}")
        
        # 默认返回 False（继续循环）
        logger.warning("Could not determine accept status, continuing loop")
        return False
