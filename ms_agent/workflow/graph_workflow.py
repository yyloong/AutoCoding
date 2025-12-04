# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections import defaultdict

from ms_agent.agent.loader import AgentLoader
from ms_agent.utils import get_logger
from ms_agent.workflow.base import Workflow
from omegaconf import DictConfig

logger = get_logger()


class GraphWorkflow(Workflow):
    """A workflow implementation that executes tasks in a sequential chain."""

    WORKFLOW_NAME = "GraphWorkflow"

    def build_workflow(self):
        if not self.config:
            return
        self.next_tasks_map = defaultdict(list)
        self.tasks_descriptions_map = {}
        self.tasks_count = defaultdict(int)
        self.start_task = getattr(self.config, "start_task", None)
        if self.start_task is not None:
            logger.info(f"GraphWorkflow self.start_task is set to {self.start_task}")
        else:
            self.start_task = list(self.config.keys())[0]
            logger.info(
                f"GraphWorkflow self.start_task is not set, use the first task {self.start_task} as self.start_task"
            )
        for task_name, task_config in self.config.items():
            if "next" in task_config:
                next_tasks = task_config["next"]
                if isinstance(next_tasks, str):
                    self.next_tasks_map[task_name] = [next_tasks]
                else:
                    self.next_tasks_map[task_name] = next_tasks
            if "description" in task_config:
                self.next_tasks_map[task_name] = self.next_tasks_map[task_name]
                self.tasks_descriptions_map[task_name] = task_config["description"]
            else:
                self.tasks_descriptions_map[task_name] = ""

        self.end_task = getattr(self.config, "end_task", None)
        if self.end_task is not None:
            logger.info(f"GraphWorkflow self.end_task is set to {self.end_task}")
        else:
            for task_name in self.config.keys():
                if len(self.next_tasks_map[task_name]) == 0:
                    self.end_task = task_name
                    logger.info(
                        f"GraphWorkflow self.end_task is not set, use the task {self.end_task} as self.end_task"
                    )
                    break

        # test if there is a path from self.start_task to end_task
        visited = set()

        def dfs(task):
            if task == self.end_task:
                return True
            visited.add(task)
            for next_task in self.next_tasks_map[task]:
                if next_task not in visited:
                    if dfs(next_task):
                        return True
            return False

        if not dfs(self.start_task):
            raise ValueError(
                f"No path from self.start_task {self.start_task} to end_task {self.end_task}"
            )

    async def run(self, inputs, **kwargs):
        agent_config = None
        task = self.start_task
        while True:
            task_info = getattr(self.config, task)
            config = getattr(task_info, "agent_config", agent_config)
            if not hasattr(task_info, "agent"):
                task_info.agent = DictConfig({})
            init_args = getattr(task_info.agent, "kwargs", {})
            init_args.pop("trust_remote_code", None)
            init_args["trust_remote_code"] = self.trust_remote_code
            init_args["mcp_server_file"] = self.mcp_server_file
            init_args["task"] = task
            init_args["load_cache"] = self.load_cache
            init_args["next_tasks"] = self.next_tasks_map.get(task, [])
            init_args["tasks_descriptions_map"] = self.tasks_descriptions_map
            if isinstance(config, str):
                init_args["config_dir_or_id"] = os.path.join(
                    self.config.local_dir, config
                )
            else:
                init_args["config"] = config
            init_args["env"] = self.env
            if "tag" not in init_args:
                init_args["tag"] = task + f"_{self.tasks_count[task]}"
            engine = AgentLoader.build(**init_args)
            await engine.run(inputs)
            await engine.exit_state()
            self.tasks_count[task] += 1
            task = engine.next_flow()
            import pdb
            pdb.set_trace()
            if task == self.end_task:
                break
        return inputs
