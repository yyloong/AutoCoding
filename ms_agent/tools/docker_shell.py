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

MAX_OUTPUT_LINES = 1000  # 防止输出过多
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
                tty=False,
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

        cmd = " ".join(cmd) if isinstance(cmd, list) else cmd

        def _exec_blocking():
            # 使用流式输出
            exec_result = self.session_container.exec_run(
                cmd=["/bin/bash", "-lc", cmd],
                stdout=True,
                stderr=True,
                tty=False,
                stream=True,
            )

            chunks = []
            # 关键：迭代 exec_result.output，而不是 exec_result 本身
            for chunk in exec_result.output:
                if not chunk:
                    continue
                # chunk 是 bytes
                text = chunk.decode("utf-8", errors="replace")
                print(text, end="", flush=True)  # 实时打印
                chunks.append(text)

            # 返回完整输出字符串
            return "".join(chunks)

        try:
            # 这里拿到的就是完整输出的字符串
            output = await asyncio.to_thread(_exec_blocking)

            # 下面保留原来的截断 + 保存逻辑，但不再 decode
            lines = output.splitlines(keepends=True)
            if len(lines) > MAX_OUTPUT_LINES:
                lines = lines[-MAX_OUTPUT_LINES:]
            joined = "".join(lines)
            if len(joined) > MAX_OUTPUT_CHARS:
                joined = joined[-MAX_OUTPUT_CHARS:]
            if len(output) > len(joined):
                full_output_path = os.path.join(self.output_dir, "full_docker_output.txt")
                with open(full_output_path, "w", encoding="utf-8") as f:
                    f.write(output)
                joined = (
                    f"\n[Output truncated to last {MAX_OUTPUT_LINES} lines or "
                    f"{MAX_OUTPUT_CHARS} characters. Full ouput have been saved to "
                    f"{full_output_path}, You can use shell tools like 'head', 'tail', "
                    f"or 'less' to view parts of the file if needed.]\n\n"
                ) + joined
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
    

if __name__ == "__main__":
    import asyncio

    # 最小配置 stub，满足 DockerBaseTool 的依赖
    class _DummyTools:
        docker_shell = {}

    class _DummyConfig:
        output_dir = "./output"   # 会被挂载到容器的 /workspace
        tools = _DummyTools()

    async def _test_docker_shell():
        print("[Test] 初始化 docker_shell 工具...")
        tool = docker_shell(_DummyConfig(), trust_remote_code=True)
        await tool.connect()

        # 在容器里简单跑两条命令：打印一句话 + 列一下 /workspace
        cmd = "cd /workspace && echo 'hello from docker_shell' && pwd && ls"
        print(f"[Test] 执行命令: {cmd}")
        result = await tool.execute_bash(cmd)

        print("\n[Test] 工具返回的最终汇总输出（截断后）：")
        print("--------------------------------------------------")
        print(result)
        print("--------------------------------------------------")

        # 清理 session 容器
        await tool.cleanup()
        print("[Test] 清理完成。")

    asyncio.run(_test_docker_shell())