# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from ms_agent.llm import Message
from omegaconf import DictConfig

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
                
                print('='*100)
                print(agent.config)
                print('='*100)

                outputs = await agent.run(inputs)
                logger.info(f'Task {task} completed')

                print('='*20 + ' OUTPUT ' + '='*20)
                print(outputs)
                print('='*50)
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
                    
                    print('='*100)
                    print(agent.config)
                    print('='*100)

                    outputs = await agent.run(inputs)
                    logger.info(f'Task {task_name} completed')

                    print('='*20 + ' OUTPUT ' + '='*20)
                    print(outputs)
                    print('='*50)
                    
                    # 如果是 breaker 任务（refiner），检查是否应该退出循环
                    if task_info.get('breaker', False):
                        if self._should_break_loop(outputs):
                            logger.info('Code accepted by refiner, exiting loop.')
                            break
                        else:
                            logger.info('Code not accepted, continuing to next iteration.')
                            # 清理 feedback.md 以便下次循环使用新的反馈
                            # （实际上 refiner 会重新生成 feedback.md）
                
                # 退出 itertools.cycle
                break
        
        return inputs

    def _inject_plan_and_feedback(self, inputs):
        """
        在 coder 运行前，读取 plan.md 和 report.md（如果存在）并注入到输入中。
        
        Args:
            inputs: 用户的原始输入（字符串或消息列表）
            
        Returns:
            增强后的输入
        """
        output_dir = self.config.get('output_dir', 'output')
        plan_path = os.path.join(output_dir, 'plan.md')
        report_path = os.path.join(output_dir, 'report.md')
        
        additional_context = "\n\n" + "="*50 + "\n"
        additional_context += "## 开发计划和反馈信息\n\n"
        
        # 读取 plan.md
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
        
        # 读取 report.md（如果存在）
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                additional_context += "### 上一轮审查反馈 (report.md):\n\n"
                additional_context += report_content + "\n\n"
                additional_context += "**注意**: 请优先修复以上审查反馈中的问题，然后继续完成未完成的功能。\n\n"
                logger.info(f"Injected report.md into coder input")
            except Exception as e:
                logger.warning(f"Failed to read report.md: {e}")
        else:
            logger.info("No report.md found (first iteration or previous iteration passed)")
        
        additional_context += "="*50 + "\n"
        
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

    def _should_break_loop(self, outputs):
        """
        判断是否应该退出循环，通过检查 refiner 的 report.md 中的 accept 字段。
        
        Args:
            outputs: refiner agent 的输出消息列表
            
        Returns:
            bool: True 表示应该退出循环（accept=true），False 表示继续循环
        """
        # 尝试从输出消息中找到 accept 字段
        # 方法1: 从消息内容中解析
        for message in reversed(outputs):
            if hasattr(message, 'content') and message.content:
                content = message.content
                
                # 查找 markdown 中的 accept 字段
                # 格式: **accept**: [true/false]
                if '**accept**:' in content.lower() or 'accept:' in content.lower():
                    # 尝试提取 accept 的值
                    import re
                    # 匹配 **accept**: true 或 accept: true
                    match = re.search(r'\*\*accept\*\*:\s*(true|false)', content, re.IGNORECASE)
                    if not match:
                        match = re.search(r'accept:\s*(true|false)', content, re.IGNORECASE)
                    
                    if match:
                        accept_value = match.group(1).lower()
                        logger.info(f"Found accept field in message: {accept_value}")
                        return accept_value == 'true'
        
        # 方法2: 从 report.md 文件中读取
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

