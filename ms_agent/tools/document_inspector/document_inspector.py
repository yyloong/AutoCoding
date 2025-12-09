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
        self.length_limit_file_endwith = config.get(
            "constraints_file_endwith", [".xlsx", ".csv", ".tsv", ".fasta", ".obo"]
        )
        self.length_limit_file_length = config.get("constraints_file_length", 1000)

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
                    "document_path": {
                        "type": "string",
                        "description": "The path to the document to be inspected.",
                    },
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of queries to be inspected in depth. Each should be a concrete question.",
                    },
                },
                "required": ["document_path"],
            },
        )
        return {"document_inspector_tool": [document_inspector_tool]}

    async def call_tool(self, server_name, *, tool_name, tool_args):
        return await getattr(self, tool_name)(**tool_args)

    async def inspect_document(self, document_path: str, query: list = []) -> str:
        document_path = os.path.join(self.output_dir, document_path)
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Input file not found: {document_path}")
        parser = SingleFileParser(
            cfg={
                "path": os.path.join(
                    os.getcwd(),
                    "ms_agent",
                    "tools",
                    "document_inspector",
                    "parser_cache",
                )
            }
        )
        input_params = {"url": document_path}
        file_parser_result = parser.call(input_params)
        if any(
            document_path.endswith(suffix) for suffix in self.length_limit_file_endwith
        ):
            if len(file_parser_result) > self.length_limit_file_length:
                file_parser_result = file_parser_result[: self.length_limit_file_length]
                return (
                    f"\n\n Document content:\n{file_parser_result}\n\n"
                    + "(Note: The following content has been truncated due to length constraints.)"
                )
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
            if 100000 > total_added_tokens > 10000:
                config = OmegaConf.load(
                    os.path.join(os.path.dirname(__file__), "inspector.yaml")
                )
                config["output_dir"] = self.output_dir
                if query == []:
                    query = "Provide a concise summary of the document:"
                else:
                    combine_query = ",".join(query)
                combine_query = f"{combine_query}\n\nDocument Content:\n{file_parser_result}"
                trust_remote_code = getattr(config, "trust_remote_code", False)
                from ms_agent.agent.llm_agent import LLMAgent

                agent = LLMAgent(
                    config=config,
                    trust_remote_code=trust_remote_code,
                    tag="document_inspector_tool",
                )
                message = await agent.run(combine_query)
                assert (
                    message[-1].role == "tool"
                    and message[-1].name == "exit_task---exit_task"
                ), "document_inspector tool did not exit properly."
                if query == []:
                    return f"The file is large,here is the summary:{message[-1].content},for more details, please use query parameter to get more information."
                else:
                    return message[-1].content
            if total_added_tokens >= 100000:
                return "The document is too large to be processed."

        except Exception as e:
            print(f"Warning: Failed to count tokens. Error: {e}")
