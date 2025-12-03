from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger

logger = get_logger()

class exit_task(ToolBase):
    """A code execution tool.
    """
    def __init__(self, config,**kwargs):
        super().__init__(config)
    
    async def get_tools(self):
        tools = {
            "exit_task": [
                Tool(
                    tool_name="exit_task",
                    server_name="exit_task",
                    description="If you ensure the task is completed, please use this tool to exit the task. ",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                ),
            ]
        }
        return tools

    async def call_tool(self, server_name, *, tool_name, tool_args):
        return await getattr(self, tool_name)(**tool_args)
        
    async def exit_task(self) -> str:
        logger.info("Exiting the task as requested.")
        return "Task exited successfully."