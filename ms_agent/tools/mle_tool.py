# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import asyncio
import os
import shutil
from typing import Optional
import subprocess
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric
from mledojo.gym.env import KaggleEnvironment
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
import time
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()
score_all_path = "/home/u-longyy/ms-agent/score_set"

class env_tools(ToolBase):
    """Kaggle related tools.
    """
    def __init__(self, config,**kwargs):
        super().__init__(config)
        tools_config = config.get("tools", {})
        tools_config = tools_config.get("env_tools", {})
        self.competition_name = tools_config.get("competition_name", "")
        os.makedirs(score_all_path, exist_ok=True)
        self.score_file_path = os.path.join(score_all_path,f"{self.competition_name}_score.txt")
        self.data_dir = tools_config.get("data_dir", "")
        self.output_dir = tools_config.get("output_dir", DEFAULT_OUTPUT_DIR)
        self.limited_step_num = tools_config.get("limited_step_num", 15)
        self.start_time = time.time()
        self.time_limit = config.get("time_limit", 3600)
        self.reflection = """The results of your previous action:
            #### Results Start ####
            {observation}
            #### Results End ####

            You still have {num_actions} actions available.
            You still have {time_left} seconds left.
            Optimize your Model/Parameters/Data Processing/Algorithm for continuous improvement.

            Output your next action strictly following Response format requirements.
        """
        self.error = """Execution failed, details below:

            #### Error Start ####
            {observation}
            #### Error End ####
            
            You still have {num_actions} actions available.
            You still have {time_left} seconds left.

            Output your next action strictly following Response format requirements.
        """
        self.step_num = 0
    
    async def connect(self):
        logger.info("Kaggle Environment connected.")
        self.registry = CompetitionRegistry()
        self.registry.register(
            name=self.competition_name,
            data_dir=self.data_dir,  # "random-acts-of-pizza/data"
            comp_info=CompInfo(
                category="General",
                level="beginner",
                output_type="submission.csv",
                higher_is_better=True
            ),
            metric_class=get_metric(self.competition_name)
        )

        self.env = KaggleEnvironment.make(
            competition_name=self.competition_name,      
            output_dir=self.output_dir,         
            competition_registry=self.registry,                  
            score_mode="position",              
            gpu_device=0,                     
            gpu_memory_limit=32,                   
            execution_timeout=3600             
        )
    
    async def get_tools(self):
        tools = {
            "env_tools": [
                Tool(
                    tool_name="request_info",
                    server_name="env_tools",
                    description="Retrieve specific competition information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "info_type": {
                                "type": "string",
                                "description": "info_type (str), must be one of: \"overview\", \"sample_submission\", \"data_structure\", \"data_path\", \"output_path\",and return information you requested",
                            },
                        },
                        "required": ["info_type"],
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    tool_name="validate_code",
                    server_name="env_tools",
                    description="Only support shell code,Test (partial) code execution for debugging purposes or \"print\" information in the output,feedback:execution result (success or failure), error message if failed, code output if success",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "code snippet to be executed",
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    tool_name="execute_code",
                    server_name="env_tools",
                    description="Only support shell code,Run completed code, generate submission and get evaluation,feedback:execution result, submission status, evaluation score",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "code snippet to be executed",
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                )
            ]
        }

        return {
            "env_tools": [
                t
                for t in tools["env_tools"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(self, server_name, *, tool_name, tool_args):
        if self.step_num >= self.limited_step_num:
            return "You have reached the maximum number of steps allowed.Please exit your task"
        try:
            obs , reward = self.env.step(tool_name, **tool_args)
            logger.info(f"Step {self.step_num}: Environment step executed. Reward: {reward}")
            self.step_num += 1
            feedback = obs.get("feedback").get("base").get("feedback")
            status = obs.get("action_status")
            prompt_template = self.error if status == "FAILED" else self.reflection
            formatted_prompt = prompt_template.format(
                observation=feedback, 
                num_actions=self.limited_step_num - self.step_num, 
                time_left=self.time_limit - (time.time() - self.start_time)
            )
            if tool_name == "execute_code":
                if os.path.exists(self.score_file_path):
                    with open(self.score_file_path, "r") as f:
                        score = float(f.read().strip())
                else:
                    score = 0.0
                now_score = obs.get("best_position_score")
                now_score = 0.0 if now_score is None else now_score
                if now_score > score:
                    with open(self.score_file_path, "w") as f:
                        f.write(str(now_score))
                    logger.info(f"New best score: {now_score}")
            return formatted_prompt
        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            return f"Error during environment step: {e}"