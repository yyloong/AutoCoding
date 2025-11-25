# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import asyncio
from typing import List

from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.openai import OpenAIChat
from ms_agent.tools.search.search_base import SearchEngine
from ms_agent.tools.search_engine import get_web_search_tool
from ms_agent.workflow.deep_research.principle import MECEPrinciple
from ms_agent.workflow.deep_research.research_workflow import ResearchWorkflow
from ms_agent.workflow.deep_research.research_workflow_beta import ResearchWorkflowBeta
from ms_agent.llm.utils import Message
from omegaconf import DictConfig


class DeepResearchCallback(Callback):
    """Callback for integrating deep_research functionality.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.research_config = config.get("tools", {}).get("deep_research", {})
        
    async def on_tool_call(self, runtime: Runtime, messages: List[Message]):
        """Handle tool calls for deep research - now handled by the tool manager."""
        # 工具调用现在由tool_manager处理，这里可以保留一些回调逻辑（如代理设置）或者留空
        pass

    async def _run_deep_research(self, user_prompt: str, args: dict) -> str:
        """This method is no longer used as the tool is now handled by the tool manager."""
        return "This method is deprecated. Use the DeepResearch tool through the tool manager."