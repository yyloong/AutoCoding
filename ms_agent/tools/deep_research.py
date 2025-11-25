# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
from typing import List, Optional

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.tools.search.search_base import SearchEngine
from ms_agent.tools.search_engine import get_web_search_tool
from ms_agent.workflow.deep_research.research_workflow_beta import ResearchWorkflowBeta
from ms_agent.llm.openai import OpenAIChat
from ms_agent.utils import get_logger

logger = get_logger()


class DeepResearch(ToolBase):
    """A deep research tool."""

    def __init__(self, config, **kwargs):
        super(DeepResearch, self).__init__(config)
        self.exclude_func(getattr(config.tools, "DeepResearch", None))
        self.trust_remote_code = kwargs.get("trust_remote_code", False)

    async def get_tools(self):
        tools = {
            "DeepResearch": [
                Tool(
                    tool_name="run_research",
                    server_name="DeepResearch",
                    description="Run deep research workflow to generate comprehensive research reports.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The research prompt or question to investigate."
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Research depth level (default: 2)",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "breadth": {
                                "type": "integer",
                                "description": "Research breadth level (default: 3)",
                                "minimum": 1,
                                "maximum": 10
                            },
                            "use_ray": {
                                "type": "boolean",
                                "description": "Whether to use Ray for parallel processing (default: false)"
                            }
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    }
                )
            ]
        }
        return {
            "DeepResearch": [
                t
                for t in tools["DeepResearch"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(
        self, server_name: str, *, tool_name: str, tool_args: dict
    ) -> str:
        return await getattr(self, tool_name)(**tool_args)

    async def run_research(self, prompt: str, depth: Optional[int] = 2, 
                          breadth: Optional[int] = 3, use_ray: Optional[bool] = False) -> str:
        """Run the deep research workflow."""
        try:
            # Get configuration from self.config
            research_config = getattr(self.config.tools, "deep_research", {})
            task_dir = research_config.get('output_dir', './output/deep_research_output')
            research_depth = depth or research_config.get('research_depth', 2)
            research_breadth = breadth or research_config.get('research_breadth', 3)
            use_ray = use_ray or research_config.get('use_ray', False)
            
            # Ensure output directory exists
            os.makedirs(task_dir, exist_ok=True)
            
            # Initialize components - 直接读取环境变量
            chat_client = OpenAIChat(
                api_key=os.getenv('OPENAI_API_KEY'),  # 直接使用环境变量OPENAI_API_KEY
                base_url=os.getenv('OPENAI_BASE_URL'),  # 直接使用环境变量OPENAI_BASE_URL
                model=os.getenv('DEEP_RESEARCH_MODEL', 'qwen3-max'),
            )
            
            search_engine = get_web_search_tool(config_file=os.getenv('DEEP_RESEARCH_CONFIG', 'conf.yaml'))
            
            # Run research workflow
            research_workflow = ResearchWorkflowBeta(
                client=chat_client,
                search_engine=search_engine,
                workdir=task_dir,
                use_ray=use_ray,
                enable_multimodal=True
            )
            
            # Execute research
            result = await research_workflow.run(
                user_prompt=prompt,
                breadth=research_breadth,
                depth=research_depth,
                is_report=True,
                show_progress=True
            )
            
            return f"Research completed successfully. Results saved to {task_dir}"
            
        except Exception as e:
            logger.error(f"Error during deep research: {str(e)}")
            return f"Error during deep research: {str(e)}"