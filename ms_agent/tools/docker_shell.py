# Copyright (c) Nanjing University
import asyncio
import os
import docker
from docker.errors import DockerException
from typing import List, Union
import subprocess
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

MAX_OUTPUT_LINES = 500  # 建议缩小，防止输出过多
MAX_OUTPUT_CHARS = 4000  # 限制最大字符数

class DockerBaseTool(ToolBase):
    """
    Base class for tools that execute commands inside Docker containers.
    Handles configuration, resource limits, and common execution logic.
    """
    def __init__(self, config, tool_name_key: str, **kwargs):
        super().__init__(config)
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        
        # Ensure output_dir is absolute for Docker mounting
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(self.output_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")

        # Get specific tool config or empty dict
        tool_config = getattr(config.tools, tool_name_key, {})
        
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.allow_read_all_files = tool_config.get("allow_read_all_files", False)
        if not self.trust_remote_code:
            self.allow_read_all_files = False
        
        # Images
        self.image = "my-dev-env:latest"

        # Initialize Docker Client
        try:
            self.client = docker.from_env()
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.client = None

    async def connect(self):
        if self.client is None:
            logger.error(f"Docker client is not connected. {self.__class__.__name__} will fail.")
        else:
            logger.info(f"{self.__class__.__name__} connected to Docker daemon.")

    async def _execute_in_docker(self, cmd: Union[str, List[str]], workdir: str = "/workspace") -> str:
        """
        Shared method to execute a command in a Docker container with resource limits and streaming output.
        """
        if not self.client:
            return "Error: Docker client not initialized."

        # Get current user ID and group ID to avoid permission issues
        current_uid = os.getuid()
        current_gid = os.getgid()

        # Helper to run blocking docker calls in a thread
        def _run_container_blocking():
            # Check/Pull Image
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling image {self.image}...")
                self.client.images.pull(self.image)

            # Convert command to list if string (use sh -c for complex strings)
            final_cmd = cmd
            if isinstance(cmd, str):
                final_cmd = ["/bin/bash", "-c", cmd]

            container = self.client.containers.run(
                self.image,
                command=final_cmd,
                volumes={self.output_dir: {'bind': workdir, 'mode': 'rw'}},
                working_dir=workdir,
                user=f"{current_uid}:{current_gid}",  # 添加用户映射，避免权限问题
                environment={
                    "HOME": workdir  # 设置工作目录为 HOME 环境变量
                },
                detach=True,
                stdout=True,
                stderr=True,
                auto_remove=False
            )
            return container

        try:
            container = await asyncio.to_thread(_run_container_blocking)
            
            # Stream logs function
            def _stream_logs():
                chunk_output = []
                for line in container.logs(stream=True, follow=True):
                    decoded = line.decode('utf-8', errors='replace').strip()
                    print(decoded) # Real-time print to host console
                    chunk_output.append(decoded + "\n")
                # 截断输出行数
                if len(chunk_output) > MAX_OUTPUT_LINES:
                    chunk_output = chunk_output[-MAX_OUTPUT_LINES:]
                joined = "".join(chunk_output)
                # 截断输出字符数
                if len(joined) > MAX_OUTPUT_CHARS:
                    joined = joined[-MAX_OUTPUT_CHARS:]
                return [joined]

            captured_output = await asyncio.to_thread(_stream_logs)
            
            # Wait for exit code
            wait_result = await asyncio.to_thread(container.wait)
            exit_code = wait_result.get('StatusCode', 0)
            
            await asyncio.to_thread(container.remove, force=True)
            
            full_output = "".join(captured_output)
            
            if exit_code != 0:
                raise subprocess.CalledProcessError(exit_code, cmd, output=full_output)
            
            return full_output if full_output else "Execution successful (no output)."

        except subprocess.CalledProcessError as e:
            return f"Command failed (Exit Code {e.returncode}):\n{e.output}"
        except Exception as e:
            return f"Docker execution system error: {str(e)}"

class docker_shell(DockerBaseTool):
    """A tool for executing arbitrary shell commands inside Docker containers."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, "docker_shell", **kwargs)
        self.exclude_func(getattr(config.tools, "docker_shell", None))

    async def get_tools(self):
        tools = {
            "docker_shell": [
                Tool(
                    tool_name="execute_bash",
                    server_name="docker_shell",
                    description="Execute a bash command in a secure Docker container. Use this for running shell commands, scripts, or system operations.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute (e.g., 'ls -la', 'pip list', 'cat file.txt').",
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    tool_name="execute_script",
                    server_name="docker_shell",
                    description="Execute a multi-line shell script in Docker. Useful for complex operations with multiple commands.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "Multi-line shell script content to execute.",
                            }
                        },
                        "required": ["script"],
                        "additionalProperties": False,
                    },
                ),
            ]
        }
        return {
            "docker_shell": [
                t for t in tools["docker_shell"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict) -> str:
        logger.info(f"Executing {tool_name} in Docker with context {self.output_dir}")
        try:
            result = await getattr(self, tool_name)(**tool_args)
        except Exception as e:
            result = f"System error: {str(e)}"
        return result

    async def execute_bash(self, command: str) -> str:
        """
        Execute a bash command in Docker.
        
        Args:
            command: The bash command to execute
            
        Returns:
            Command output or error message
        """
        logger.info(f"Executing bash command in {self.image}: {command}")
        return await self._execute_in_docker(command)

    async def execute_script(self, script: str) -> str:
        """
        Execute a multi-line shell script in Docker.
        
        Args:
            script: Multi-line shell script content
            
        Returns:
            Script output or error message
        """
        logger.info(f"Executing shell script in {self.image}")
        return await self._execute_in_docker(script)