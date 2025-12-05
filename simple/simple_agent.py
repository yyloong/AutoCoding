import sys
from copy import deepcopy
from typing import List, AsyncGenerator, Any

from ms_agent import LLMAgent
from ms_agent.llm import Message


class DockerAgent(LLMAgent):

    # __init__

    async def step(
        self, messages: List[Message]
    ) -> AsyncGenerator[List[Message], Any]:  # type: ignore
        """
        A agent that can execute commands in Docker containers.

        Exit the current task by calling the tool `exit_task---exit_task`.
        """
        messages = deepcopy(messages)
        if (not self.load_cache) or messages[-1].role != 'assistant':
            messages = await self.condense_memory(messages)
            await self.on_generate_response(messages)
            tools = await self.tool_manager.get_tools()

            if self.stream:
                self.log_output('[assistant]:')
                _content = ''
                is_first = True
                _response_message = None
                for _response_message in self.llm.generate(
                        messages, tools=tools):
                    if is_first:
                        messages.append(_response_message)
                        is_first = False
                    new_content = _response_message.content[len(_content):]
                    sys.stdout.write(new_content)
                    sys.stdout.flush()
                    _content = _response_message.content
                    messages[-1] = _response_message
                    yield messages
                sys.stdout.write('\n')
            else:
                _response_message = self.llm.generate(messages, tools=tools)
                if _response_message.content:
                    self.log_output('[assistant]:')
                    self.log_output(_response_message.content)

            # Response generated
            self.handle_new_response(messages, _response_message)
            await self.on_tool_call(messages)
        else:
            # Set load_cache to `false` to avoid affect later operations
            self.load_cache = False
            # Meaning the latest message is `assistant`, this prevents a different response if there are sub-tasks.
            _response_message = messages[-1]
        self.save_history(messages)

        if _response_message.tool_calls:
            messages = await self.parallel_tool_call(messages)

        if _response_message.tool_calls and _response_message.tool_calls[-1]["tool_name"] == "exit_task---exit_task":
            self.runtime.should_stop = True

        await self.after_tool_call(messages)
        self.log_output(
            f'[usage] prompt_tokens: {_response_message.prompt_tokens}, '
            f'completion_tokens: {_response_message.completion_tokens}')
        yield messages