import os
import shutil
from typing import Optional
from omegaconf import DictConfig
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from omegaconf import OmegaConf
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

logger = get_logger()

class DeepresearchTool(ToolBase):
    """DeepResearch Tool"""

    name = "deepresearch_tool"
    description = (
        "A tool for deep research tasks. "
        "Useful for conducting in-depth analysis and research on complex topics."
    )

    def __init__(self, config, **kwargs):
        super().__init__(config)

    async def connect(self):
        """Connect to DeepResearch model"""
        logger.info("DeepResearch model connected.")

    async def get_tools(self):
        """Get DeepResearch tool"""
        deepresearch_tool = Tool(
            tool_name="research",
            server_name="deepresearch_tool",
            description="Use this tool to conduct in-depth research on the given request.",
            parameters={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The request to be researched in depth.",
                    },
                },
                "required": ["request"],
            },
        )
        return {"deepresearch_tool": [deepresearch_tool]}

    async def call_tool(self, server_name, *, tool_name, tool_args):
        return await getattr(self, tool_name)(**tool_args)

    async def research(self, request: str) -> str:
        config = OmegaConf.load(os.path.join(
            os.path.dirname(__file__), "research.yaml"
        ))
        trust_remote_code = getattr(config, "trust_remote_code", False)
        from ms_agent.agent.llm_agent import LLMAgent
        agent = LLMAgent(
            config=config,
            trust_remote_code=trust_remote_code,
            tag="deepresearch_tool",
        )
        message = await agent.run(request)
        assert (
            message[-1].role == "tool"
            and message[-1].name == "finish---exit_task"
        ), "DeepResearch tool did not exit properly."
        return message[-1].content
