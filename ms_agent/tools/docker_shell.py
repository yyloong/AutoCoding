# Copyright (c) Nanjing University
import asyncio
import os
import docker
from docker.errors import DockerException
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

MAX_OUTPUT_LINES = 1000  # 建议缩小，防止输出过多
MAX_OUTPUT_CHARS = 10000  # 限制最大字符数

class DockerBaseTool(ToolBase):
    def __init__(self, config, tool_name_key: str, **kwargs):
        super().__init__(config)
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")

        tool_config = getattr(config.tools, tool_name_key, {})
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.allow_read_all_files = tool_config.get("allow_read_all_files", False)
        if not self.trust_remote_code:
            self.allow_read_all_files = False

        self.image = "my-dev-env:latest"

        try:
            self.client = docker.from_env()
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.client = None

        # 新增：为该工具实例维护一个长期容器
        self.session_container = None

    async def connect(self):
        if self.client is None:
            logger.error(f"Docker client is not connected. {self.__class__.__name__} will fail.")
        else:
            logger.info(f"{self.__class__.__name__} connected to Docker daemon.")
            # 在 connect 时创建 session 容器
            await self._ensure_session_container()

    async def _ensure_session_container(self, workdir: str = "/workspace"):
        """
        Lazily create a long-lived bash container for this tool instance.
        """
        if self.session_container is not None:
            # 简单检查一下容器是否还活着
            try:
                self.session_container.reload()
                if self.session_container.status in ("running", "created"):
                    return
            except Exception:
                self.session_container = None

        if not self.client:
            logger.error("Docker client not initialized, cannot create session container.")
            return

        current_uid = os.getuid()
        current_gid = os.getgid()

        def _run_container_blocking():
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling image {self.image}...")
                self.client.images.pull(self.image)

            container = self.client.containers.run(
                self.image,
                command=["/bin/bash"],   # 长期 bash 会话
                volumes={self.output_dir: {'bind': workdir, 'mode': 'rw'}},
                working_dir=workdir,
                user=f"{current_uid}:{current_gid}",
                environment={"HOME": workdir},
                detach=True,
                tty=True,                # 伪终端
                stdin_open=True,
                stdout=True,
                stderr=True,
                auto_remove=False,
            )
            return container

        self.session_container = await asyncio.to_thread(_run_container_blocking)
        logger.info(f"Started session container {self.session_container.id} for {self.__class__.__name__}")

    async def _exec_in_session(self, cmd: str, workdir: str = "/workspace") -> str:
        """
        Execute a command inside the long-lived session container.
        """
        if not self.client:
            return "Error: Docker client not initialized."

        await self._ensure_session_container(workdir=workdir)
        if self.session_container is None:
            return "Error: Failed to create session container."

        # 在 bash 中执行命令，保持同一会话上下文（cd/环境变量等）
        cmd = " ".join(cmd) if isinstance(cmd, list) else cmd

        def _exec_blocking():
            exec_result = self.session_container.exec_run(
                cmd=["/bin/bash", "-lc", cmd],
                stdout=True,
                stderr=True,
                tty=True,
            )
            return exec_result

        try:
            exec_result = await asyncio.to_thread(_exec_blocking)
            output = exec_result.output.decode("utf-8", errors="replace")
            # 这里仍然做输出截断
            lines = output.splitlines(keepends=True)
            if len(lines) > MAX_OUTPUT_LINES:
                lines = lines[-MAX_OUTPUT_LINES:]
            joined = "".join(lines)
            if len(joined) > MAX_OUTPUT_CHARS:
                joined = joined[-MAX_OUTPUT_CHARS:]
            if len(output) > len(joined):
                joined = f"\n[Output truncated to last {MAX_OUTPUT_LINES} lines or {MAX_OUTPUT_CHARS} characters]\n" + joined
            return joined if joined.strip() else "Execution successful (no output)."
        except Exception as e:
            return f"Docker execution system error: {str(e)}"

    async def cleanup(self):
        """
        Optional: cleanup session container when tool is destroyed.
        """
        if self.session_container is not None:
            try:
                self.session_container.stop()
            except Exception:
                pass
            try:
                self.session_container.remove(force=True)
            except Exception:
                pass
            self.session_container = None

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
        return await self._exec_in_session(command, workdir="/workspace")

    async def execute_script(self, script: str) -> str:
        """
        Execute a multi-line shell script in Docker.
        
        Args:
            script: Multi-line shell script content
            
        Returns:
            Script output or error message
        """
        logger.info(f"Executing shell script in {self.image}")
        return await self._exec_in_session(script, workdir="/workspace")