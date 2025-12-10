from ms_agent.rag.ensemble_rag import MultiLlamaIndexRAG
from ms_agent.config import Config
import asyncio

if __name__ == "__main__":
    config = Config.from_task("ms_agent/rag/model_config.yaml")
    rag = MultiLlamaIndexRAG(config)
    asyncio.run(rag.initialize_all_components())
    asyncio.run(rag.save_index())
    answer = asyncio.run(rag.query("best algorithm and code for Spaceship Titanic competition"))
    print(answer)
    asyncio.run(rag.llama_debug_info())
