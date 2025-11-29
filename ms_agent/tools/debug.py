# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
import re
import docker
from docker.types import DeviceRequest
from docker.errors import DockerException
from typing import Optional, List, Union
import subprocess
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

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

        # Get specific tool config or empty dict
        tool_config = getattr(config.tools, tool_name_key, {})
        
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.allow_read_all_files = tool_config.get("allow_read_all_files", False)
        if not self.trust_remote_code:
            self.allow_read_all_files = False

        # Resource limits
        self.cpu_limit = tool_config.get("cpu_limit", 1.0)
        self.gpu_limit = tool_config.get("gpu_limit", 0)
        self.memory_limit = tool_config.get("memory_limit", "1g")
        
        # Images
        self.python_image = tool_config.get("python_image", "python:3.10-slim")
        self.node_image = tool_config.get("node_image", "node:18-slim")

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

    async def _execute_in_docker(self, image: str, cmd: Union[str, List[str]], workdir: str = "/workspace") -> str:
        """
        Shared method to execute a command in a Docker container with resource limits and streaming output.
        """
        if not self.client:
            return "Error: Docker client not initialized."

        # Configure GPU requests
        device_requests = []
        if self.gpu_limit:
            count = -1 if self.gpu_limit == "all" else int(self.gpu_limit)
            device_requests.append(DeviceRequest(count=count, capabilities=[['gpu']]))

        # Get current user ID and group ID to avoid permission issues
        current_uid = os.getuid()
        current_gid = os.getgid()

        # Helper to run blocking docker calls in a thread
        def _run_container_blocking():
            # Check/Pull Image
            try:
                self.client.images.get(image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling image {image}...")
                self.client.images.pull(image)

            # Convert command to list if string (use sh -c for complex strings)
            final_cmd = cmd
            if isinstance(cmd, str):
                final_cmd = ["/bin/sh", "-c", cmd]

            container = self.client.containers.run(
                image,
                command=final_cmd,
                volumes={self.output_dir: {'bind': workdir, 'mode': 'rw'}},
                working_dir=workdir,
                user=f"{current_uid}:{current_gid}",  # 添加用户映射，避免权限问题
                nano_cpus=int(self.cpu_limit * 1e9),
                mem_limit=self.memory_limit,
                device_requests=device_requests,
                detach=True,
                stdout=True,
                stderr=True,
                auto_remove=False
            )
            return container

        try:
            container = await asyncio.to_thread(_run_container_blocking)
            
            captured_output = []

            # Stream logs function
            def _stream_logs():
                chunk_output = []
                for line in container.logs(stream=True, follow=True):
                    decoded = line.decode('utf-8', errors='replace').strip()
                    print(decoded) # Real-time print to host console
                    chunk_output.append(decoded + "\n")
                return chunk_output

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


class Environment_set_up(DockerBaseTool):
    """A environment set up tool running in Docker."""

    def __init__(self, config, **kwargs):
        super().__init__(config, "environment_set_up", **kwargs)
        self.exclude_func(getattr(config.tools, "environment_set_up", None))
        self.uv_init_done = False

    async def get_tools(self):
        tools = {
            "environment_set_up": [
                Tool(
                    tool_name="npm_install",
                    server_name="environment_set_up",
                    description="Install Node.js project dependencies using npm inside a Docker container.",
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
                    description="Run command with uv to set up python project environment inside Docker.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command_content": {
                                "type": "string",
                                "description": "uv command to run (e.g., 'venv', 'pip install requests').",
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
                t for t in tools["environment_set_up"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict) -> str:
        # No need to os.chdir locally, Docker bind mount handles the directory context
        logger.info(f"Executing {tool_name} in Docker with context {self.output_dir}")
        try:
            result = await getattr(self, tool_name)(**tool_args)
        except Exception as e:
            result = f"System error: {str(e)}"
        return result

    async def npm_install(self) -> str:
        """Install Node.js project dependencies using npm in Docker."""
        return await self._execute_in_docker(
            self.node_image, 
            ["npm", "install"]
        )

    async def uv_command(self, command_content: str) -> str:
        """Install Python project dependencies using uv in Docker."""
        
        # Logic to check state (mapped to local file existence since volume is shared)
        venv_path_host = os.path.join(self.output_dir, ".venv", "bin", "activate")
        
        # If .venv exists on host, we consider init done
        if os.path.exists(venv_path_host):
            self.uv_init_done = True

        if not self.uv_init_done:
            if re.match(r"venv\s+--python\s+", command_content) is None:
                return "please run 'uv venv --python <python_version>' first."
        
        # Validate commands
        if (
            not self.uv_init_done 
            and re.match(r"venv\s+--python\s+", command_content) is None
        ):
             pass # Logic handled above, but keeping structure
        elif (
            self.uv_init_done
            and re.match(r"pip\s+install\s+", command_content) is None
            and re.match(r"pip\s+uninstall\s+", command_content) is None
            and re.match(r"venv", command_content) is None # Allow re-init
        ):
             return "now only support 'venv', 'pip uninstall xxx' and 'pip install xxx' commands."

        # Construct the command. 
        # Note: Standard python images do not have 'uv'. We try to ensure it exists.
        # We wrap in sh -c to allow chaining commands.
        
        setup_uv = "pip install uv"
        uv_cmd = f"uv {command_content}"
        
        # If initialized, target the venv python
        if self.uv_init_done and "venv" not in command_content:
             # Force uv to use the venv we created
             uv_cmd = f"uv {command_content} --python .venv/bin/python"

        full_cmd = f"{setup_uv} && {uv_cmd}"
        
        result = await self._execute_in_docker(self.python_image, full_cmd)
        
        # Update state after execution
        if os.path.exists(venv_path_host):
            self.uv_init_done = True
            
        return result


class run_code(DockerBaseTool):
    """A code execution tool running inside Docker."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, "run_code", **kwargs)

    async def get_tools(self):
        tools = {
            "run_code": [
                Tool(
                    tool_name="run_file",
                    server_name="run_code",
                    description="Execute a file in a secure Docker container.",
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
                                "description": "The programming language."
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
        logger.info(f"Running code in Docker. Output Dir: {self.output_dir}")
        try:
            result = await getattr(self, tool_name)(**tool_args)
        except Exception as e:
            result = f"System error: {str(e)}"
        return result
        
    async def run_file(self, file: str, language: str) -> str:
        cmd = []
        image = ""
        
        if language == "python":
            image = self.python_image
            # Check if venv exists in the mounted directory structure
            # Since we can't check inside the container easily before running, 
            # we check the host path which is mounted to /workspace
            venv_python = os.path.join(self.output_dir, ".venv", "bin", "python")
            
            if os.path.exists(venv_python):
                # Use the venv python inside the container
                cmd = [".venv/bin/python", file]
            else:
                # Fallback to system python
                cmd = ["python", file]
                
        elif language == "javascript":
            image = self.node_image
            cmd = ["node", file]
        else:
            return f"Unsupported language: {language}"

        logger.info(f"Running {language} file '{file}' in {image}")
        return await self._execute_in_docker(image, cmd)


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
                            },
                            "image": {
                                "type": "string",
                                "description": "Docker image to use. Defaults to 'python:3.10-slim'. Use 'node:18-slim' for Node.js commands.",
                                "default": "python:3.10-slim"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Command timeout in seconds. Default is 300 seconds (5 minutes).",
                                "default": 300
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
                            },
                            "image": {
                                "type": "string",
                                "description": "Docker image to use. Defaults to 'python:3.10-slim'.",
                                "default": "python:3.10-slim"
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

    async def execute_bash(self, command: str, image: str = "python:3.10-slim", timeout: int = 300) -> str:
        """
        Execute a bash command in Docker.
        
        Args:
            command: The bash command to execute
            image: Docker image to use
            timeout: Command timeout in seconds
            
        Returns:
            Command output or error message
        """
        logger.info(f"Executing bash command in {image}: {command}")
        
        # Check if venv exists and should be activated for python commands
        venv_activate = os.path.join(self.output_dir, ".venv", "bin", "activate")
        
        # If using python image and venv exists, prepend activation
        if "python" in image.lower() and os.path.exists(venv_activate):
            # Use bash to allow source command
            full_command = f"bash -c 'source .venv/bin/activate && {command}'"
        else:
            full_command = command
            
        return await self._execute_in_docker(image, full_command)

    async def execute_script(self, script: str, image: str = "python:3.10-slim") -> str:
        """
        Execute a multi-line shell script in Docker.
        
        Args:
            script: Multi-line shell script content
            image: Docker image to use
            
        Returns:
            Script output or error message
        """
        logger.info(f"Executing shell script in {image}")
        
        # Check if venv exists
        venv_activate = os.path.join(self.output_dir, ".venv", "bin", "activate")
        
        # Prepend venv activation if exists and using python image
        if "python" in image.lower() and os.path.exists(venv_activate):
            script = f"source .venv/bin/activate\n{script}"
        
        # Wrap script in bash -c for proper execution
        full_command = f"bash -c {repr(script)}"
        
        return await self._execute_in_docker(image, full_command)