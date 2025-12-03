import asyncio
import sys

from simple.simple_agent import SimpleAgent
from ms_agent.config import Config

async def run_query(query: str):
    config = Config.from_task('simple')
    config.llm.modelscope_api_key = 'sk-d67a35829268468a8e864369c7540fe7'
    engine = SimpleAgent(config=config)

    _content = ''
    generator = await engine.run(query, stream=True)
    async for _response_message in generator:
        new_content = _response_message[-1].content[len(_content):]
        sys.stdout.write(new_content)
        sys.stdout.flush()
        _content = _response_message[-1].content
    sys.stdout.write('\n')
    return _content


if __name__ == '__main__':
    query = '请先在当前工作目录下创建虚拟环境，然后安装相应库，然后请实际创建 pie_chart.py 文件并运行它.'
    asyncio.run(run_query(query))