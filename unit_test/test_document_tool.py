from omegaconf import OmegaConf
from ms_agent.tools.document_inspector.document_inspector import document_inspector
import asyncio


if __name__ == "__main__":
    config = OmegaConf.load("ms_agent/tools/document_inspector/inspector.yaml")
    config["output_dir"] = "./unit_test"
    document_inspector_tool = document_inspector(config)
    asyncio.run(
        document_inspector_tool.inspect_document(
            document_path="paper.pdf",
            query=[
                "describe the pseudocode of Algorithm 2 Refining the DRL Agent",
                "how many methods are there in figure 'SAC Agent Refining Performance in Hopper Game',what are their trends and performance?",
            ],
        )
    )
