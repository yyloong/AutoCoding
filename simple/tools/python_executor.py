import sys
import io
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from omegaconf import DictConfig

class PythonExecutorTool(ToolBase):
    """A tool for executing Python code."""

    def __init__(self, config):
        super(PythonExecutorTool, self).__init__(config)
        self.exclude_func(getattr(config.tools, 'python_executor', None))

    async def get_tools(self):
        tools = {
            'python_executor': [
                Tool(
                    tool_name='execute_code',
                    server_name='python_executor',
                    description='Execute Python code and return the result.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'code': {
                                'type': 'string',
                                'description': 'Python code to execute.',
                            }
                        },
                        'required': ['code'],
                        'additionalProperties': False
                    }
                ),
            ]
        }
        return {
            'python_executor': [
                t for t in tools['python_executor']
                if t['tool_name'] not in self.exclude_functions
            ]
        }

    async def execute_code(self, code: str) -> str:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        mystdout = io.StringIO()
        sys.stdout = mystdout
        sys.stderr = mystdout
        local_vars = {}

        print("="*80)
        print("Executing code:")
        print(code)
        print("="*80)

        try:
            lines = code.strip().split('\n')
            if not lines:
                return "No code to execute."
            
            *body, last = lines
            # 执行前面的语句
            if body:
                exec('\n'.join(body), {}, local_vars)
            
            # 尝试 eval 最后一行，如果失败则 exec
            try:
                result = eval(last, {}, local_vars)
                if result is not None:
                    print(result)
            except Exception:
                exec(last, {}, local_vars)
            
            output = mystdout.getvalue()
            # 关键修复：确保不返回 None
            return output.strip() if output else "Executed successfully (no output)."
            
        except Exception as e:
            output = mystdout.getvalue()
            return (output.strip() + f"\nError: {e}") if output else f"Error: {e}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    async def call_tool(self, server_name: str, tool_name: str, tool_args: dict) -> str:
        if tool_name == 'execute_code':
            code = tool_args.get('code', '')
            return await self.execute_code(code)
        else:
            return f'Unknown tool: {tool_name}'