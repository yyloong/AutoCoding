# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List
from copy import deepcopy

import os
import json
from file_parser import extract_code_blocks
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class ArtifactCallback(Callback):
    """Save the output code to local disk.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)
        self.files_json = None
        self.agent_type = config.get("agent", "")

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_generate_response(self, runtime: Runtime,
                                   messages: List[Message]):
        for message in messages:
            if message.role == 'assistant' and message.tool_calls and not message.content:
                # Claude seems does not allow empty content
                message.content = 'I should do a tool calling to continue:\n'

    async def on_tool_call(self, runtime: Runtime, messages: List[Message]):
        if not messages[-1].tool_calls or messages[-1].role != 'assistant':
            return
        results = []
        tool_calls = deepcopy(messages[-1].tool_calls)
        for tc in tool_calls:
            if tc['tool_name'] == 'web_research---search' or tc['tool_name'] == 'web_research---visit_page':
                if self.agent_type == "research":
                    os.environ["http_proxy"] = os.environ.get("shorttime_http_proxy","")
                    os.environ["https_proxy"] = os.environ.get("shorttime_https_proxy","")
                
            if tc['tool_name'] == 'file_system---write_file':
                await self.file_system.create_directory()
                path = json.loads(tc['arguments']).get('path', '')
                if self.agent_type == "research" and path != "analysis.md":
                    logger.warning(
                        f"Only analysis.md can be written in planning agent.Skipped writing file {path}.")
                    results.append(
                        f"Only analysis.md can be written in planning agent.Skipped writing file {path}.")
                    messages[-1].tool_calls.remove(tc)
                    continue

                if self.agent_type == "architecture" and path != "files.json":
                    logger.warning(
                        f"Only files.json can be written in architecture agent.Skipped writing file {path}.")
                    results.append(
                        f"Only files.json can be written in architecture agent.Skipped writing file {path}.")
                    messages[-1].tool_calls.remove(tc)
                    continue

                if not await self.check_file_right(path):
                    logger.warning(
                        f"File {path} is not allowed to be written.Skipped,Please check files.json for the file paths.The path must be strictly the same.")
                    results.append(
                        f"File {path} is not allowed to be written.Skipped,Please check files.json for the file paths.The path must be strictly the same.")
                    messages[-1].tool_calls.remove(tc)
                    continue

        r = '\n'.join(results)
        if len(r) > 0:
            messages.append(Message(role='user', content=r))
        #import pdb
        #pdb.set_trace()
    
    async def check_file_right(self,filename: str) -> bool:

        if not os.path.exists(os.path.join(self.file_system.output_dir, "files.json")):
            print(f"try to create new file {self.file_system.output_dir} {filename}")
            return True
        if self.files_json is None:
            with open(os.path.join(self.file_system.output_dir, "files.json"), "r") as f:
                self.files_json = json.load(f)
                
        for f in self.files_json:
            if f == filename:
                return True
        return False
    
    async def after_tool_call(self, runtime, messages):
        # reset proxy after web research tool call
        if self.agent_type == "research":
            if messages[-1].tool_calls:
                for tc in messages[-1].tool_calls:
                    if tc['tool_name'] == 'web_research---search' or tc['tool_name'] == 'web_research---visit_page':
                        del os.environ["http_proxy"] 
                        del os.environ["https_proxy"] 