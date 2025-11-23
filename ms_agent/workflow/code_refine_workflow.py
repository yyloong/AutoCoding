# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from omegaconf import DictConfig

logger = get_logger()


class DeepcodeWorkflow(Workflow):
    """A workflow implementation with loop support between refine and coding agents."""

    WORKFLOW_NAME = 'DeepcodeWorkflow'

    def build_workflow(self):
        """Build the workflow chain from config."""
        if not self.config:
            return

        # 过滤掉配置字段，只保留任务定义
        reserved_keys = {'type', 'local_dir'}  # 这些是配置字段，不是任务
        task_configs = {k: v for k, v in self.config.items() if k not in reserved_keys}

        has_next = set()
        start_task = None
        
        # Find all tasks that are referenced as 'next'
        for task_name, task_config in task_configs.items():
            if 'next' in task_config:
                next_tasks = task_config['next']
                if isinstance(next_tasks, str):
                    has_next.add(next_tasks)
                elif isinstance(next_tasks, list):
                    has_next.update(next_tasks)

        # Find start task (not referenced by any other task)
        for task_name in task_configs.keys():
            if task_name not in has_next:
                start_task = task_name
                break

        if start_task is None:
            raise ValueError('No start task found')

        # Build the workflow chain
        result = []
        current_task = start_task

        while current_task:
            result.append(current_task)
            next_task = None
            task_config = task_configs[current_task]
            if 'next' in task_config:
                next_tasks = task_config['next']
                if isinstance(next_tasks, str):
                    next_task = next_tasks
                elif isinstance(next_tasks, list):
                    next_task = next_tasks[0]

            current_task = next_task
            
        self.workflow_chains = result
        logger.info(f'Built workflow chain: {" -> ".join(result)}')

    async def run(self, inputs, **kwargs):
        """
        Execute the workflow with loop support between refine and coding.

        Args:
            inputs (Any): Initial input data for the first task.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The final output after workflow completion.
        """
        agent_config = None
        idx = 0
        step_inputs = {}
        max_iterations = kwargs.get('max_iterations', 10)  # 防止无限循环
        iteration_count = 0

        while True:
            if idx >= len(self.workflow_chains):
                break

            task = self.workflow_chains[idx]
            task_info = getattr(self.config, task)
            config = getattr(task_info, 'agent_config', agent_config)
            
            # Prepare agent initialization arguments
            if not hasattr(task_info, 'agent'):
                task_info.agent = DictConfig({})
            
            init_args = getattr(task_info.agent, 'kwargs', {})
            init_args.pop('trust_remote_code', None)
            init_args['trust_remote_code'] = self.trust_remote_code
            init_args['mcp_server_file'] = self.mcp_server_file
            init_args['task'] = task
            init_args['load_cache'] = self.load_cache
            
            if isinstance(config, str):
                init_args['config_dir_or_id'] = os.path.join(
                    self.config.local_dir, config)
            else:
                init_args['config'] = config
                
            init_args['env'] = self.env
            if 'tag' not in init_args:
                init_args['tag'] = task
            
            # Build and run agent
            engine = AgentLoader.build(**init_args)
            step_inputs[idx] = (inputs, config)
            
            logger.info(f'Running task: {task} (iteration: {iteration_count})')
            outputs = await engine.run(inputs)
            
            # Handle refine task - check if code is acceptable
            if task == 'refine':
                iteration_count += 1
                
                # 检查 refine 的输出，判断是否需要继续循环
                should_continue = self._should_continue_refining(outputs)
                
                if should_continue and iteration_count < max_iterations:
                    # 循环回 coding 阶段
                    logger.info(f'Refine suggests improvements, returning to coding (iteration {iteration_count})')
                    coding_idx = self.workflow_chains.index('coding')
                    idx = coding_idx
                    inputs = outputs
                    agent_config = engine.config
                    continue
                else:
                    if iteration_count >= max_iterations:
                        logger.warning(f'Max iterations ({max_iterations}) reached, stopping refinement loop')
                    else:
                        logger.info('Refine approved, workflow completed')
                    # 退出循环
                    return outputs
            
            # Normal flow progression
            next_idx = engine.next_flow(idx)
            
            if next_idx == idx + 1:
                inputs = outputs
                agent_config = engine.config
            else:
                inputs, agent_config = step_inputs[next_idx]
                
            idx = next_idx

        return inputs

    def _should_continue_refining(self, refine_output):
        """
        Determine if the refine agent wants to continue refining.
        
        This method checks the refine output to see if it indicates
        the code needs further improvement.
        
        Args:
            refine_output: Output from the refine agent
            
        Returns:
            bool: True if should continue refining, False if code is acceptable
        """
        # 你需要根据 refine agent 的实际输出格式来实现这个逻辑
        # 以下是几种可能的实现方式：
        
        # 方式1: 检查输出中是否包含特定关键词
        if isinstance(refine_output, str):
            # 如果包含这些词，说明需要继续改进
            continue_keywords = ['需要改进', 'need improvement', 'issues found', 
                               'bugs', 'errors', '问题', '建议修改']
            # 如果包含这些词，说明代码没问题
            approve_keywords = ['approved', 'looks good', 'no issues', 
                              '没有问题', '通过', 'LGTM']
            
            output_lower = refine_output.lower()
            
            if any(keyword.lower() in output_lower for keyword in approve_keywords):
                return False
            if any(keyword.lower() in output_lower for keyword in continue_keywords):
                return True
        
        # 方式2: 检查是否是字典格式，包含状态字段
        if isinstance(refine_output, dict):
            if 'status' in refine_output:
                return refine_output['status'] != 'approved'
            if 'needs_refinement' in refine_output:
                return refine_output['needs_refinement']
            if 'issues' in refine_output:
                return len(refine_output['issues']) > 0
        
        # 默认：如果不确定，继续循环（保守策略）
        logger.warning('Unable to determine refine status, continuing refinement')
        return True