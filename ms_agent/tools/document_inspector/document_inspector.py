import os
import tiktoken
from typing import Optional
from omegaconf import DictConfig
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from omegaconf import OmegaConf
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR
from ms_agent.utils.file_parser_utils.file_parser import SingleFileParser

logger = get_logger()

class document_inspector(ToolBase):
    """document_inspector Tool"""

    name = "document_inspector_tool"
    description = (
        "A tool for document inspection tasks. "
        "Useful for conducting in-depth analysis and research on complex topics."
    )

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.output_dir = config.get("output_dir", DEFAULT_OUTPUT_DIR)

    async def connect(self):
        """Connect to DeepResearch model"""
        logger.info("DeepResearch model connected.")

    async def get_tools(self):
        """Get document_inspector tool"""
        document_inspector_tool = Tool(
            tool_name="inspect_document",
            server_name="document_inspector_tool",
            description="Reads non-standard documents (PDF, DOCX, etc.). If the file is small,it will return the full content.If 'query' is not provided and the file is large,it will return a summary of it.If 'query' is provided and the file is large, it will research and return an answer.",
            parameters={
                "type": "object",
                "properties": {
                    "document_path":{
                        "type": "string",
                        "description": "The path to the document to be inspected.",
                    },
                    "request": {
                        "type": "string",
                        "description": "The request to be inspected in depth.",
                    },
                },
                "required": ["request"],
            },
        )
        return {"document_inspector_tool": [document_inspector_tool]}

    async def call_tool(self, server_name, *, tool_name, tool_args):
        return await getattr(self, tool_name)(**tool_args)

    async def inspect_document(self, document_path: str, request: str) -> str:
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Input file not found: {document_path}")
        parser = SingleFileParser(cfg={
            'path': os.path.join(os.getcwd(),'workspace','parser_cache')
        })
        input_params = {"url": document_path}
        file_parser_result = parser.call(input_params)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            file_tokens = len(encoding.encode(file_parser_result))
            prefix_text = "\n\n Document content:\n"
            prefix_tokens = len(encoding.encode(prefix_text))
            total_added_tokens = file_tokens + prefix_tokens
    
            print(f"-------- Token Statistics --------")
            print(f"File Path: {document_path}")
            print(f"Content Length (Chars): {len(file_parser_result)}")
            print(f"Token Count (Est.): {file_tokens}")
            print(f"Total Added Tokens: {total_added_tokens}")
            print(f"----------------------------------")
            if total_added_tokens <= 10000:
                return f"{prefix_text}{file_parser_result}\n\n"
            if total_added_tokens > 10000:
                config = OmegaConf.load(os.path.join(
                    os.path.dirname(__file__), "inspector.yaml"
                ))
                config['output_dir'] = self.output_dir
                if request.strip() == "":
                    request = "Provide a concise summary of the document:"
                request = f"{request}\n\nDocument Content:\n{file_parser_result}"
                trust_remote_code = getattr(config, "trust_remote_code", False)
                from ms_agent.agent.llm_agent import LLMAgent
                agent = LLMAgent(
                    config=config,
                    trust_remote_code=trust_remote_code,
                    tag="document_inspector_tool",
                )
                message = await agent.run(request)
                assert (
                    message[-1].role == "tool"
                    and message[-1].name == "exit_task---exit_task"
                ), "document_inspector tool did not exit properly."
                return message[-1].content

        except Exception as e:
            print(f"Warning: Failed to count tokens. Error: {e}")
        
