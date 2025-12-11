from omegaconf import OmegaConf
from ms_agent.tools.deepresearch_tool.deepresearch import DeepresearchTool
import asyncio

if __name__ == "__main__":
    config = OmegaConf.load("ms_agent/tools/deepresearch_tool/research.yaml")
    deepresearch_tool = DeepresearchTool(config)
    asyncio.run(
        deepresearch_tool.research(
            request="How many projects are there in the github repository modelscope/ms-agent? "
        )
    )
