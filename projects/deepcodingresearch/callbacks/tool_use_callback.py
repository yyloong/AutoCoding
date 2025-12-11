from typing import List
from copy import deepcopy

import os
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class Tool_Use_Callback(Callback):
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
            if tc['tool_name'] == 'kaggle_tools---download_dataset' or tc['tool_name'] == 'kaggle_tools---submit_csv':
                os.environ["http_proxy"] = os.environ.get("shorttime_http_proxy","")
                os.environ["https_proxy"] = os.environ.get("shorttime_https_proxy","")
            
    async def after_tool_call(self, runtime, messages):
        if messages[-1].tool_calls:
            for tc in messages[-1].tool_calls:
                if tc['tool_name'] == 'kaggle_tools---download_dataset' or tc['tool_name'] == 'kaggle_tools---submit_csv':
                    logger.info("Resetting proxy settings after kaggle_tools usage.")
                    os.environ["http_proxy"] = ""
                    os.environ["https_proxy"] = ""
        
        