# Copyright (c) Nanjing University.
import re
import subprocess
import os
import shutil
from typing import Optional
from typing import List, Union

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR

from ms_agent.tools.web_search_tools.search import Search
from ms_agent.tools.web_search_tools.visit_page import Visit_page

logger = get_logger()


class Web_research(ToolBase):
    """A web research tool."""

    def __init__(self, config, **kwargs):
        super(Web_research, self).__init__(config)
        self.exclude_func(getattr(config.tools, "Web_research", None))
        self.output_dir = getattr(config, "output_dir", DEFAULT_OUTPUT_DIR)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.allow_read_all_files = getattr(
            getattr(config.tools, "Web_research", {}),
            "allow_read_all_files",
            False,
        )
        if not self.trust_remote_code:
            self.allow_read_all_files = False

        self.uv_init_done = False

    async def get_tools(self):
        tools = {
            "Web_research": [
                Tool(
                    tool_name="visit_page",
                    server_name="Web_research",
                    description="Visit webpage(s) and return the summary of the content.",
                    parameters = {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "array",
                                "minItems": 1,
                                "description": "Array of webpage URLs to visit.Each element is a string,for example ['http://example.com/page1','http://example.com/page2']",
                            },
                            "goal": {
                                    "type": "string",
                                    "description": "The goal of the visit for webpage(s)."
                            }
                        },
                        "required": ["url", "goal"],
                        "additionalProperties": False,
                    }
                ),
                Tool(
                    tool_name="search",
                    server_name="Web_research",
                    description="Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
                    parameters = {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                            "description": "Array of query strings. Include multiple complementary search queries in a single call."
                            },
                        },
                        "required": ["query"],
                    }
                ),
            ]
        }
        return {
            "Web_research": [
                t
                for t in tools["Web_research"]
                if t["tool_name"] not in self.exclude_functions
            ]
        }

    async def call_tool(
        self, server_name: str, *, tool_name: str, tool_args: dict
    ) -> str:
        return await getattr(self, tool_name)(**tool_args)

    async def search(self, query: str) -> str:
        params = {"query": query}
        return Search.call_tool(params)
    
    async def visit_page(self, url: Optional[Union[str, List[str]]]=None, goal: Optional[str]=None) -> str:
        params = {
            "url": url,
            "goal": goal
        }
        Visit_page.url = getattr(self.config["llm"], "openai_base_url", None)
        Visit_page.api_key = os.environ.get("OPENAI_API_KEY")
        Visit_page.model = getattr(self.config["llm"], "model", None)
        return await Visit_page.call_tool(params)
   