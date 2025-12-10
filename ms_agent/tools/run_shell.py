# Copyright (c) Alibaba, Inc. and its affiliates.
import subprocess
import time
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
            output_file_path = None # 用于标记是否转为文件输出

            # 实时读取输出
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                decoded_line = line.decode(errors='replace') 
                line_len = len(decoded_line)

                # 模式 A: 已经转为文件输出模式
                if output_file_path:
                    with open(output_file_path, "a", encoding="utf-8") as f:
                        f.write(decoded_line)
                
                # 模式 B: 还在内存捕获模式，检查是否超出限制
                else:
                    if total_chars + line_len > MAX_OUTPUT_CHARS:
                        # --- 触发限制：切换到文件模式 ---
                        
                        # 1. 生成文件名 (例如: output_1715000000.txt)
                        filename = f"cmd_output_{int(time.time())}.txt"
                        output_file_path = os.path.join(os.getcwd(), filename) # 或者指定特定的 temp 目录
                        
                        # 2. 将内存中已有的内容写入文件
                        with open(output_file_path, "w", encoding="utf-8") as f:
                            f.write("".join(captured_output)) # 写入之前的历史
                            f.write(decoded_line)             # 写入当前这行
                        
                        # 3. 清空内存以释放资源
                        captured_output = [] 
                        
                        msg = f"\n[SYSTEM] Output exceeded {MAX_OUTPUT_CHARS} chars. Redirecting remaining output to file: {filename}"
                        print(msg) # 控制台提示
                        
                        # 注意：这里不再 terminate 进程，而是继续循环直到进程结束，但后续内容都会进 if output_file_path 分支
                    else:
                        # 未超限，正常存入内存并打印
                        print(decoded_line.strip())
                        captured_output.append(decoded_line)
                        total_chars += line_len

            # 等待进程退出
            return_code = await process.wait()

            # --- 5. 构造返回结果 ---
            
            # 情况 A: 输出被转存到了文件
            if output_file_path:
                status_msg = "Success" if return_code == 0 else f"Failed (Code {return_code})"
                return (
                    f"{status_msg}. Output length exceeded limit ({MAX_OUTPUT_CHARS} chars).\n"
                    f"Full output has been saved to: {output_file_path}\n"
                    f"It is advised that you don't read the entire file at once to avoid context overload."
                    f" You can use shell tools like 'head', 'tail', or 'less' to view parts of the file."
                )

            # 情况 B: 输出未超限，直接返回字符串
            else:
                full_output = "".join(captured_output)
                if return_code != 0:
                    return f"Command failed (Code {return_code}):\n{full_output}"
                
                return full_output if full_output else "Success (No output)."

        except Exception as e:
            return f"Execution error: {str(e)}"