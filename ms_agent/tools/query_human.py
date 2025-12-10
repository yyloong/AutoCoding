# Copyright (c) Nanjing University.
from typing import Dict, Any
import time
import sys
import select

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()


class Query_human(ToolBase):
    """A tool to ask the human user for information via console input."""

    def __init__(self, config, **kwargs):
        super(Query_human, self).__init__(config)
        # 支持在配置里排除某些函数，和其它工具保持一致
        self.exclude_func(getattr(config.tools, "Query_human", None))
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        self.timeout = 30

    async def get_tools(self):
        tools = {
            "Query_human": [
                Tool(
                    tool_name="query_human",
                    server_name="Query_human",
                    description=(
                        "Ask the human user for information or feedback via "
                        "console input. Use this when you need clarification, "
                        "preferences, or decisions from the human."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": (
                                    "The question, request for opinion, or "
                                    "message to display to the human user."
                                ),
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                ),
            ]
        }

        return {
            "Query_human": [
                t
                for t in tools["Query_human"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(
        self, server_name: str, *, tool_name: str, tool_args: Dict[str, Any]
    ) -> str:
        # 和其它工具保持一致：根据 tool_name 分发到对应方法
        return await getattr(self, tool_name)(**tool_args)

    async def query_human(self, prompt: str) -> str:
        """真正和人类交互的函数，使用 input() 获取反馈。"""

        banner = r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                   ║    
║    ██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ ███╗   ██╗    ██╗███╗   ██╗██████╗ ██╗   ██╗████████╗      ║
║    ██║  ██║██║   ██║████╗ ████║██╔══██╗████╗  ██║    ██║████╗  ██║██╔══██╗██║   ██║╚══██╔══╝      ║
║    ███████║██║   ██║██╔████╔██║███████║██╔██╗ ██║    ██║██╔██╗ ██║██████╔╝██║   ██║   ██║         ║   
║    ██╔══██║██║   ██║██║╚██╔╝██║██╔══██║██║╚██╗██║    ██║██║╚██╗██║██╔═══╝ ██║   ██║   ██║         ║
║    ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║    ██║██║ ╚████║██║     ╚██████╔╝   ██║         ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝    ╚═╝         ║    
║                                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

        # ANSI 颜色：紫色 + 加粗
        PURPLE = "\033[38;2;148;102;194m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        # 打印显眼的提示
        print(PURPLE + BOLD + banner + RESET)
        print(PURPLE + "The agent needs your help:" + RESET)
        print()
        print(BOLD + prompt + RESET)
        print()
        print(PURPLE + "-" * 100 + RESET)

        # 超过 30 s 没有输入则跳过,并返回相应信息


        print(PURPLE + "You have 30 seconds to answer..." + RESET)
        sys.stdout.flush()
        answer = ""
        try:
            # select 只在类 Unix 系统下可用
            ready, _, _ = select.select([sys.stdin], [], [], self.timeout)
            if ready:
                answer = sys.stdin.readline().rstrip("\n")
            else:
                print(PURPLE + f"No input received in {self.timeout} seconds, skipping..." + RESET)
            answer = "[No human feedback received, maybe they are busy. You need to continue.]" if answer == "" else answer
        except Exception as e:
            # 在某些非交互环境下防止崩溃
            logger.warning("Exception while waiting for input: %s", e)
            answer = "[Input error or timeout]"

        logger.info("Human answer: %s", answer)
        return answer