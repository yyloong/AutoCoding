from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger

logger = get_logger()


class State_transition(ToolBase):
    """A code execution tool."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.next_states = config.get("next_tasks", [])
        self.states_descriptions = config.get("tasks_descriptions", {})
        assert len(self.next_states) > 0, "The next_states must be non-empty."

    async def get_tools(self):
        tools = {
            "state_transition": [
                Tool(
                    tool_name=self.next_states[i],
                    server_name="state_transition",
                    description=self.states_descriptions.get(self.next_states[i], ""),
                    parameters={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Detailed description about your purpose for transitioning to this state.If you want to ask for other state for help,please describe your request very clear.If you just finish your work,you can summary what you have done.",
                            },
                        },
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                )
                for i in range(len(self.next_states))
            ]
        }
        return tools

    async def call_tool(self, server_name, *, tool_name, tool_args):
        logger.info(f"Transitioning to state: {tool_name} with message: {tool_args}")
        tools = await self.get_tools()
        for tool in tools.get(server_name, []):
            if tool["tool_name"] == tool_name:
                return "Successful"
        return "failed, state not found"
