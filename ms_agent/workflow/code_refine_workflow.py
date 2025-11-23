# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from omegaconf import DictConfig, OmegaConf

logger = get_logger()


class DeepcodeWorkflow(Workflow):
    """A workflow implementation with loop support between refine and coding agents."""

    WORKFLOW_NAME = 'DeepcodeWorkflow'

    def build_workflow(self):
        """Build the workflow chain from config."""
        if not self.config:
            return

        # 过滤掉配置字段，只保留任务定义
        reserved_keys = {'type', 'local_dir', 'name', 'tools', 'callbacks', 
                        'llm', 'generation_config', 'prompt', 'agent', 
                        'max_chat_round', 'tool_call_timeout', 'output_dir', 'help'}
        
        task_configs = {k: v for k, v in self.config.items() 
                       if k not in reserved_keys and isinstance(v, (dict, DictConfig))}

        logger.info(f'Task configs: {list(task_configs.keys())}')

        has_next = set()
        start_task = None
        
        # Find all tasks that are referenced as 'next'
        for task_name, task_config in task_configs.items():
            # 确保 task_config 是 DictConfig
            if not isinstance(task_config, (dict, DictConfig)):
                logger.warning(f'Task {task_name} config is not a dict: {type(task_config)}')
                continue
            
            logger.info(f'Processing task: {task_name}, config type: {type(task_config)}')
            
            # 使用 OmegaConf 的方式访问字段
            if 'next' in task_config or hasattr(task_config, 'next'):
                next_tasks = task_config.get('next') if isinstance(task_config, dict) else getattr(task_config, 'next', None)
                
                if next_tasks is not None:
                    logger.info(f'  next tasks: {next_tasks} (type: {type(next_tasks)})')
                    
                    # 处理不同类型的 next
                    if isinstance(next_tasks, str):
                        has_next.add(next_tasks)
                    elif isinstance(next_tasks, (list, tuple)):
                        has_next.update(next_tasks)
                    else:
                        # 可能是 ListConfig
                        try:
                            has_next.update(list(next_tasks))
                        except:
                            logger.warning(f'Cannot process next_tasks: {next_tasks}')

        logger.info(f'Tasks referenced as next: {has_next}')

        # Find start task (not referenced by any other task)
        for task_name in task_configs.keys():
            if task_name not in has_next:
                start_task = task_name
                break

        if start_task is None:
            raise ValueError('No start task found')

        logger.info(f'Start task: {start_task}')

        # Build the workflow chain
        result = []
        current_task = start_task
        visited = set()

        while current_task and current_task not in visited:
            result.append(current_task)
            visited.add(current_task)
            
            next_task = None
            if current_task in task_configs:
                task_config = task_configs[current_task]
                
                # 使用统一的方式获取 next
                if 'next' in task_config or hasattr(task_config, 'next'):
                    next_tasks = task_config.get('next') if isinstance(task_config, dict) else getattr(task_config, 'next', None)
                    
                    if next_tasks is not None:
                        if isinstance(next_tasks, str):
                            next_task = next_tasks
                        elif isinstance(next_tasks, (list, tuple)) and len(next_tasks) > 0:
                            next_task = next_tasks[0]
                        else:
                            try:
                                next_list = list(next_tasks)
                                if len(next_list) > 0:
                                    next_task = next_list[0]
                            except:
                                pass
                        
                        logger.info(f'Task {current_task} -> next: {next_task}')
            
            current_task = next_task
            
        self.workflow_chains = result
        logger.info(f'Built workflow chain: {" -> ".join(result)}')

    async def run(self, inputs, **kwargs):
        """
        Execute the workflow with loop support between refine and coding.
        """
        agent_config = None
        idx = 0
        step_inputs = {}
        max_iterations = kwargs.get('max_iterations', 10)
        iteration_count = 0

        while idx < len(self.workflow_chains):
            task = self.workflow_chains[idx]
            task_info = getattr(self.config, task)
            config = getattr(task_info, 'agent_config', agent_config)
            
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
            
            logger.info(f'Running task: {task} (iteration: {iteration_count}/{max_iterations})')
            engine = AgentLoader.build(**init_args)
            step_inputs[idx] = (inputs, config)
            
            outputs = await engine.run(inputs)
            logger.info(f'Task {task} completed')
            
            if task == 'refine':
                iteration_count += 1
                should_continue = self._should_continue_refining(outputs)
                
                if should_continue and iteration_count < max_iterations:
                    logger.info(f'Refine suggests improvements, returning to coding (iteration {iteration_count})')
                    coding_idx = self.workflow_chains.index('coding')
                    idx = coding_idx
                    inputs = outputs
                    agent_config = engine.config
                    continue
                else:
                    if iteration_count >= max_iterations:
                        logger.warning(f'Max iterations ({max_iterations}) reached')
                    else:
                        logger.info('Refine approved, workflow completed')
                    return outputs
            
            inputs = outputs
            agent_config = config
            idx += 1

        return inputs

    def _should_continue_refining(self, refine_output):
        """
        Determine if refining should continue by checking if exit_task was called.
        
        Args:
            refine_output: Output from the refine agent
            
        Returns:
            bool: True if should continue refining, False if exit_task was called
        """
        # 优先检查是否调用了 exit_task 工具
        tool_calls = refine_output.get('tool_calls', [])
        if tool_calls:
            for call in tool_calls:
                tool_name = call.get('tool_name', '')
                if 'exit_task' in tool_name:
                    logger.info('Refine agent called exit_task, stopping refinement loop')
                    return False