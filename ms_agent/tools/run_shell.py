# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import asyncio
import os
import shlex
import re
from typing import Optional, List

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

# --- 配置常量 ---

# 1. 限制输出的最大字符数 (约 2000 tokens)
# 如果输出超过此长度，将强制截断并终止进程，防止撑爆上下文
MAX_OUTPUT_CHARS = 8000 

# 2. 命令行直接禁止的命令
FORBIDDEN_CLI_COMMANDS = {
    "sudo", "su", "shutdown", "reboot", 
    "mkfs", "dd", "fdisk", "mount", "umount", "chmod", "chown",
    "ssh", "scp", "mv", "cp", "top", "vi", "vim", "nano" # 增加交互式工具
}

# 3. 脚本内容扫描：禁止出现在 .sh 文件内部的关键字（正则匹配）
FORBIDDEN_SCRIPT_PATTERNS = [
    r"\brm\b",           # 删除文件
    r"\bsudo\b",         # 提权
    r"\bsu\b",           # 切换用户
    r"\bdd\b",           # 磁盘操作
    r"\bmkfs",           # 格式化
    r"\bwget\b",         # 下载
    r"\bcurl\b",         # 网络请求
    r"\bshutdown\b",     # 关机
    r"\breboot\b",       # 重启
    r">\s*/dev/",        # 重定向到设备文件
    r":\(\)\s*\{",       # Fork bomb
    r"\bchmod\b",        # 修改权限
    r"\bchown\b"         # 修改所有者
]

SHELL_INTERPRETERS = {"bash", "sh", "zsh"}

class execute_shell(ToolBase):
    """A safe shell execution tool with script scanning and output truncation.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.exclude_func(getattr(config.tools, "execute_shell", None))
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        
    async def connect(self):
        logger.warning_once(
            f"[IMPORTANT] execute_shell is protected. "
            f"Output limit: {MAX_OUTPUT_CHARS} chars. "
            f"Dangerous commands and patterns in scripts are blocked."
        )
    
    async def get_tools(self):
        tools = {
            "execute_shell": [
                Tool(
                    tool_name="run_shell_command",
                    server_name="execute_shell",
                    description=f"Execute a shell command or script safely.Commands below is forbidden: {', '.join(FORBIDDEN_CLI_COMMANDS)}. Output is limited to {MAX_OUTPUT_CHARS} characters to prevent overload.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command or script to execute.Don't change the working directory.",
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
            ]
        }
        return tools

    async def call_tool(self, server_name, *, tool_name, tool_args):
        now_dir = os.getcwd()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        os.chdir(self.output_dir)
        try:
            result = await getattr(self, tool_name)(**tool_args)
        except Exception as e:
            result = f"System error: {str(e)}"
        finally:
            os.chdir(now_dir)
        return result

    def _scan_file_safety(self, file_path: str) -> Optional[str]:
        """Scans a file for dangerous patterns."""
        if not os.path.exists(file_path):
            return f"File not found for safety scan: {file_path}"
        
        try:
            # 限制读取大小，防止扫描超大文件耗尽内存
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024: # 限制 1MB
                return "Security Alert: Script file is too large to scan safely. Execution blocked."

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for pattern in FORBIDDEN_SCRIPT_PATTERNS:
                if re.search(pattern, content):
                    return f"Security Alert: Script contains forbidden pattern '{pattern}'. Execution blocked."
            return None
        except Exception as e:
            return f"Error scanning file: {str(e)}"

    async def run_shell_command(self, command: str) -> str:
        try:
            # --- 1. 参数解析 ---
            try:
                args = shlex.split(command)
            except ValueError as e:
                return f"Invalid command syntax: {str(e)}"
            
            if not args: return "Empty command."
            program = args[0]
            
            # --- 2. 识别与转换 ---
            script_file = None
            final_args = args

            if program in SHELL_INTERPRETERS:
                if len(args) < 2: return "Error: Interpreter requires a file."
                for arg in args[1:]:
                    if arg.startswith("-"):
                        return f"Error: Flags like '{arg}' not allowed."
                script_file = args[1]

            elif program.endswith(".sh") or program.startswith("./"):
                script_file = program
                final_args = ["bash", program] + args[1:]
            
            elif program in FORBIDDEN_CLI_COMMANDS:
                return f"Error: Command '{program}' is forbidden."

            # --- 3. 脚本内容安全扫描 ---
            if script_file:
                target_path = script_file
                if target_path.startswith("./"):
                    target_path = target_path[2:]
                
                scan_error = self._scan_file_safety(target_path)
                if scan_error:
                    logger.warning(f"Blocked dangerous script execution: {script_file}")
                    return scan_error

            # --- 4. 执行与输出流控制 ---
            logger.info(f"Executing: {final_args}")
            process = await asyncio.create_subprocess_exec(
                final_args[0], *final_args[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            captured_output = []
            total_chars = 0
            truncated = False

            # 实时读取输出
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                decoded_line = line.decode(errors='replace') # 这里先不strip，为了计算准确长度
                line_len = len(decoded_line)

                # 检查长度限制
                if total_chars + line_len > MAX_OUTPUT_CHARS:
                    # 计算剩余可允许的字符数
                    remaining = MAX_OUTPUT_CHARS - total_chars
                    captured_output.append(decoded_line[:remaining])
                    
                    msg = f"\n\n[SYSTEM WARNING] Output truncated. Exceeded limit of {MAX_OUTPUT_CHARS} characters."
                    captured_output.append(msg)
                    print(msg) # 打印到控制台
                    
                    truncated = True
                    # 终止进程，避免后台继续消耗资源或卡死 pipe
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    break
                
                print(decoded_line.strip())
                captured_output.append(decoded_line)
                total_chars += line_len

            # 等待进程完全退出
            if truncated:
                # 如果是我们主动截断并terminate的，不需要等待正常退出码
                await process.wait()
                return "".join(captured_output)
            else:
                return_code = await process.wait()
                full_output = "".join(captured_output)

                if return_code != 0:
                    return f"Command failed (Code {return_code}):\n{full_output}"
                
                return full_output if full_output else "Success (No output)."

        except Exception as e:
            return f"Execution error: {str(e)}"