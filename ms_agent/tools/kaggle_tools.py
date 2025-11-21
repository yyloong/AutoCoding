# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import asyncio
import os
import shutil
from typing import Optional
import subprocess

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

class kaggle_tools(ToolBase):
    """Kaggle related tools.
    """
    def __init__(self, config,**kwargs):
        super().__init__(config)
        self.output_dir = getattr(config, 'output_dir', DEFAULT_OUTPUT_DIR)
    
    async def get_tools(self):
        tools = {
            "kaggle_tools": [
                Tool(
                    tool_name="download_dataset",
                    server_name="kaggle_tools",
                    description="Download the dataset for the competition from Kaggle.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "competition": {
                                "type": "string",
                                "description": "The name of the Kaggle competition to download the dataset from.The files can only be downloaded in the current working directory.",
                            }
                        },
                        "required": ["competition"],
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    tool_name="submit_csv",
                    server_name="kaggle_tools",
                    description="Submit the solution file (csv file) to the Kaggle competition.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "competition": {
                                "type": "string",
                                "description": "The name of the Kaggle competition to submit the solution to.",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "The path to the csv file to be submitted.",
                            },
                            "submit_message": {
                                "type": "string",
                                "description": "The message for the submission.",
                            }
                        },
                        "required": ["competition", "file_path","submit_message"],
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    tool_name="get_scores",
                    server_name="kaggle_tools",
                    description="Get the scores of the submissions for the Kaggle competition.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "competition": {
                                "type": "string",
                                "description": "The name of the Kaggle competition to get the scores from.",
                            }
                        },
                        "required": ["competition"],
                        "additionalProperties": False,
                    },
                )
            ]
        }
        return tools

    async def call_tool(self, server_name, *, tool_name, tool_args):
        return await getattr(self, tool_name)(**tool_args)
        
    async def download_dataset(self, competition: str) -> str:
        cmd_set = [
            f'kaggle competitions download -c {competition} -p {self.output_dir}/{competition}',
            f'unzip {self.output_dir}/{competition}/*.zip -d {self.output_dir}/{competition}',
            f'rm {self.output_dir}/{competition}/*.zip'
        ]
        results = []
        for cmd in cmd_set:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f'Error executing command "{cmd}": {stderr.decode().strip()}')
                return f'Error executing command "{cmd}": {stderr.decode().strip()}'
            results.append(stdout.decode().strip())

        return f'Dataset for competition {competition} downloaded successfully to {self.output_dir}/{competition}.'

    
    async def submit_csv(self, competition: str, file_path: str, submit_message: str) -> str:
        cmd = f'kaggle competitions submit -c {competition} -f {file_path} -m "{submit_message}"'
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f'Error submitting file: {stderr.decode().strip()}')
            return f'Error submitting file: {stderr.decode().strip()}'
        
        logger.info(f'File {file_path} submitted successfully to competition {competition}.')
        return f'File {file_path} submitted successfully to competition {competition}. Response: {stdout.decode().strip()}'

    async def get_scores(self, competition: str) -> str:
        cmd = f'kaggle competitions submissions -c {competition}'
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f'Error getting scores: {stderr.decode().strip()}')
            return f'Error getting scores: {stderr.decode().strip()}'
        
        logger.info(f'Scores for competition {competition} retrieved successfully.')
        return f'Scores for competition {competition}:\n{stdout.decode().strip()}'