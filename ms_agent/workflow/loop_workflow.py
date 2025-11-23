# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from omegaconf import DictConfig

logger = get_logger()

class LoopWorkflow(Workflow):
    """A workflow implementation with loop support between refine and coding agents."""

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
        Execute the workflow with loop support between refine and coding.
        """
        step_inputs = {}
        max_iterations = kwargs.get('max_iterations', 10)
        iteration_count = 0
        idx = 0

        for task in self.workflow_chains:
            if task != 'loop':
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
                
                logger.info(f'Running task: {task} (iteration: {iteration_count}/{max_iterations})')
                agent = AgentLoader.build(**init_args)
                step_inputs[idx] = (inputs, config)
                
                print('='*100)
                print(agent.config)
                print('='*100)

                outputs = await agent.run(inputs)
                logger.info(f'Task {task} completed')

                print('='*20 + ' OUTPUT ' + '='*20)
                print(outputs)
                print('='*50)
                inputs = outputs
                idx += 1

            else:
                tasks = self.config[task]
                import itertools
                for task in itertools.cycle(tasks.keys()):
                    task_info = tasks[task]
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
                    
                    logger.info(f'Running task: {task} (iteration: {iteration_count}/{max_iterations})')
                    agent = AgentLoader.build(**init_args)
                    step_inputs[idx] = (inputs, config)
                    
                    print('='*100)
                    print(agent.config)
                    print('='*100)

                    outputs = await agent.run(inputs)
                    logger.info(f'Task {task} completed')

                    print('='*20 + ' OUTPUT ' + '='*20)
                    print(outputs)
                    print('='*50)
                    
                    inputs = outputs
                    if task_info['breaker']:
                        if self._should_break_loop(outputs[-2]):
                            logger.info('Exiting loop as break condition met.')
                            break
        return inputs

    def _should_break_loop(self, message):
        """
        Determine if refining should continue by checking if exit_task was called.
        
        Args:
            refine_output: Output from the refine agent
            
        Returns:
            bool: True if should continue refining, False if exit_task was called
        """
        # ```json
        # {
        # "accepted": true/false,
        # "issues": [ "错误描述1 (步骤编号)", "错误描述2 (步骤编号)" ],
        # "suggestions": "改进建议"
        # }
        # ```
        
        # 匹配是否有以上格式的 JSON 输出，并检查 accepted 字段
        if "```json" in message.content and "```" in message.content:
            try:
                json_str = message.content.split("```json")[1].split("```")[0].strip()
                import json
                output_json = json.loads(json_str)
                if 'accepted' in output_json and output_json['accepted'] is True:
                    return True
            except Exception as e:
                logger.warning(f"Failed to parse JSON from refine agent output: {e}")
        return False
