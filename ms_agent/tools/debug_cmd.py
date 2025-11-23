# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import asyncio
import os
import shutil
from typing import Optional

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

class run_code(ToolBase):
    """A code execution tool.
    """
    def __init__(self, config,**kwargs):
        super().__init__(config)
        self.exclude_func(getattr(config.tools, "run_code", None))
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.allow_read_all_files = getattr(
            getattr(config.tools, "run_code", {}),
            "allow_read_all_files",
            False,
        )
        if not self.trust_remote_code:
            self.allow_read_all_files = False
    
    async def connect(self):
        logger.warning_once(
            "[IMPORTANT]run_code is not implemented with sandbox, please consider other similar "
            "tools if you want to run dangerous code."
        )
    
    async def get_tools(self):
        tools = {
            "run_code": [
                Tool(
                    tool_name="run_file",
                    server_name="run_code",
                    description="a tool that can execute the file you provided in the parameters. ",
                    parameters={
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "The file path to be executed."
                            },
                            "language": {
                                "type": "string",
                                "enum": ["python", "javascript"],
                                "description": "The programming language of the code snippet."
                            }
                        },
                        "required": ["file", "language"],
                        "additionalProperties": False,
                    },
                ),
            ]
        }
        return tools

    async def call_tool(self, server_name, *, tool_name, tool_args):
        return await getattr(self, tool_name)(**tool_args)
        
    async def run_file(self, file: str, language: str) -> str:
        try:
            code_file_path = os.path.join(self.output_dir, file)
            
            cmd = []
            if language == "python":
                cmd = ["uv", "run", "--python=.venv/bin/python", code_file_path]
            elif language == "javascript":
                cmd = ["node", code_file_path]
            else:
                return f"Unsupported language: {language}"

            # 创建异步子进程
            process = await asyncio.create_subprocess_exec(
                cmd[0], *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT # 合并错误流
            )

            captured_output = []

            # 异步读取输出
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode().strip() # 解码字节
                print(decoded_line)           # 实时打印
                captured_output.append(decoded_line + "\n")

            # 等待进程结束
            return_code = await process.wait()
            
            full_output = "".join(captured_output)

            if return_code != 0:
                return f"Error executing code: {full_output}"

            return full_output if full_output else "Code executed successfully with no output."

        except Exception as e:
            return f"System error: {str(e)}"