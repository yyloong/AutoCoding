# Copyright (c) Alibaba, Inc. and its affiliates.
import re
import subprocess
import os
import shutil
from typing import Optional

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()


class Environment_set_up(ToolBase):
    """A environment set up tool.

    TODO: This tool now is a simple implementation, sandbox or mcp TBD.
    """

    def __init__(self, config, **kwargs):
        super(Environment_set_up, self).__init__(config)
        self.exclude_func(getattr(config.tools, "environment_set_up", None))
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.allow_read_all_files = getattr(
            getattr(config.tools, "environment_set_up", {}),
            "allow_read_all_files",
            False,
        )
        if not self.trust_remote_code:
            self.allow_read_all_files = False

        self.uv_init_done = False

    async def connect(self):
        logger.warning_once(
            "[IMPORTANT]Environment_set_up is not implemented with sandbox, please consider other similar "
            "tools if you want to run dangerous code."
        )

    async def get_tools(self):
        tools = {
            "environment_set_up": [
                Tool(
                    tool_name="npm_install",
                    server_name="environment_set_up",
                    description="Install Node.js project dependencies using npm. This command reads the package.json file in the current directory and installs all listed dependencies into the node_modules folder.",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                ),

                Tool(
                    tool_name="uv_command",
                    server_name="environment_set_up",
                    description="run command with uv to set up python project environment. Args: command_content (str): uv command to run, e.g., if you want to run 'uv venv', just pass 'venv' here,Now only support 'venv' , 'pip uninstall xxx' and 'pip install xxx' commands.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command_content": {
                                "type": "string",
                                "description": "uv command to run, e.g., if you want to run 'uv venv', just pass 'venv' here, Now only support 'venv' , 'pip uninstall xxx' and 'pip install xxx' commands.",
                            },
                        },
                        "required": ["command_content"],
                        "additionalProperties": False,
                    },
                ),
            ]
        }
        return {
            "environment_set_up": [
                t
                for t in tools["environment_set_up"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(
        self, server_name: str, *, tool_name: str, tool_args: dict
    ) -> str:
        return await getattr(self, tool_name)(**tool_args)

    async def npm_install(self) -> str:
        """Install Node.js project dependencies using npm.

        This command reads the package.json file in the current directory and
        installs all listed dependencies into the node_modules folder.
        """
        try:
            # 修改：使用 Popen 替代 run 以支持实时输出
            cmd = ["npm", "install"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout，确保 capture 且实时显示
                text=True,
                bufsize=1  # 行缓冲
            )

            captured_output = []
            # 实时读取并打印
            for line in process.stdout:
                print(line, end="", flush=True)  # 输出到终端
                captured_output.append(line)     # 存入内存

            process.wait()  # 等待子进程结束
            
            # 组合完整输出
            full_output = "".join(captured_output)

            # 检查返回码，模拟 check=True 的行为
            if process.returncode != 0:
                # 将捕获到的完整输出作为 stderr 参数传递，以便 except 块能获取到错误详情
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr=full_output)
            
            output = full_output
            
        except subprocess.CalledProcessError as e:
            output = f"npm install failed with error: {e.stderr}"
        return output

    async def uv_command(self, command_content: str) -> str:
        """Install Python project dependencies using uv.

        This command reads the pyproject.toml file in the current directory and
        installs all listed dependencies into the virtual environment.

        Args:
            command_content (str): uv command to run, e.g., if you want to run
            'uv install -r requirements.txt', just pass 'install -r requirements.txt' here.
        """
        if not self.uv_init_done:
            if re.match(r"venv\s+--python\s+", command_content) is None:
                output = "please run 'uv venv --python <python_version>' to initialize the virtual environment and follow the format strictly."
                return output
            output = await self.run_uv_command(command_content)
            uv_venv_path = os.path.join(".venv", "bin", "activate")
            if os.path.exists(uv_venv_path):
                self.uv_init_done = True
            return output

        if (
            re.match(r"pip\s+install\s+", command_content) is None
            and re.match(r"pip\s+uninstall\s+", command_content) is None
        ):
            output = "now only support 'venv','pip uninstall xxx' and 'pip install xxx' commands."
            return output

        output = await self.run_uv_command(command_content)
        return output

    async def run_uv_command(self, command_content: str) -> str:
        try:
            # 保持原有的命令构建逻辑
            if self.uv_init_done:
                command = ["uv"] + command_content.split() + ["-p", ".venv/bin/python"]
            else:
                command = ["uv"] + command_content.split()

            # 修改：使用 Popen 替代 run 以支持实时输出
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # 将 stderr 合并到 stdout
                text=True,
                bufsize=1
            )

            captured_output = []
            # 实时读取并打印
            for line in process.stdout:
                print(line, end="", flush=True)
                captured_output.append(line)

            process.wait()
            
            full_output = "".join(captured_output)

            # 模拟 check=True
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, stderr=full_output)
            
            output = full_output
            
        except subprocess.CalledProcessError as e:
            output = f"uv '{command_content}' failed with error: {e.stderr}"
        return output