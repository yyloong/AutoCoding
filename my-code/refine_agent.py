import sys
from copy import deepcopy
from typing import List, AsyncGenerator, Any, Union

from ms_agent import LLMAgent
from ms_agent.llm import Message
from ms_agent.utils.logger import logger

class RefineAgent(LLMAgent):

    async def step(
        self, messages: List[Message]
    ) -> AsyncGenerator[List[Message], Any]:
        """
        Execute a single step in the agent's interaction loop.

        This method performs the following operations in sequence:
        1. Deep copies the current message history to avoid mutation issues.
        2. Refines memory based on the current conversation state.
        3. Triggers pre-response callbacks.
        5. Generates a response from the LLM using available tools.
        6. Optionally streams the response output to stdout.
        7. Triggers post-response callbacks.
        8. Handles parallel tool calls if needed.
        9. Triggers post-tool-call callbacks.
        10. Returns the updated message history.

        The step may be retried up to two times on failure due to the `@async_retry` decorator.

        Args:
            messages (List[Message]): Current message history.

        Returns:
            List[Message]: Updated message history after this step.
        """
        messages = deepcopy(messages)
        if (not self.load_cache) or messages[-1].role != 'assistant':
            messages = await self.condense_memory(messages)
            await self.on_generate_response(messages)
            tools = await self.tool_manager.get_tools()
            # print(tools)

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
        
        # print(_response_message)
        # print(f"Tool calls: {_response_message.tool_calls}")
        if _response_message.tool_calls:
            messages = await self.parallel_tool_call(messages)
        # else:
        #     self.runtime.should_stop = True

        if _response_message and _response_message.tool_calls[-1]["tool_name"] == "exit_task---exit_task":
            self.runtime.should_stop = True

        await self.after_tool_call(messages)
        self.log_output(
            f'[usage] prompt_tokens: {_response_message.prompt_tokens}, '
            f'completion_tokens: {_response_message.completion_tokens}')
        yield messages

    async def run_loop(self, messages: Union[List[Message], str],
                       **kwargs) -> AsyncGenerator[Any, Any]:
        """Run the agent, mainly contains a llm calling and tool calling loop.

        Args:
            messages (Union[List[Message], str]): Input data for the agent. Can be a raw string prompt,
                                               or a list of previous interaction messages.
        Returns:
            List[Message]: A list of message objects representing the agent's response or interaction history.
        """
        try:
            self.max_chat_round = getattr(self.config, 'max_chat_round',
                                          LLMAgent.DEFAULT_MAX_CHAT_ROUND)
            self.register_callback_from_config()
            self.prepare_llm()
            self.prepare_runtime()
            await self.prepare_tools()
            await self.load_memory()
            await self.prepare_rag()
            self.runtime.tag = self.tag

            if messages is None:
                messages = self.query

            self.config, self.runtime, messages = self.read_history(messages)

            if self.runtime.round == 0:
                # 0 means no history
                messages = await self.create_messages(messages)
                await self.do_rag(messages)
                await self.on_task_begin(messages)

            for message in messages:
                if message.role != 'system':
                    self.log_output('[' + message.role + ']:')
                    self.log_output(message.content)
            while not self.runtime.should_stop:
                async for messages in self.step(messages):
                    yield messages
                self.runtime.round += 1
                # save memory and history
                self.save_memory(messages)
                self.save_history(messages)

                # +1 means the next round the assistant may give a conclusion
                if self.runtime.round >= self.max_chat_round + 1:
                    if not self.runtime.should_stop:
                        messages.append(
                            Message(
                                role='assistant',
                                content=
                                f'Task {messages[1].content} was cutted off, because '
                                f'max round({self.max_chat_round}) exceeded.'))
                    self.runtime.should_stop = True
                    yield messages

            # save memory
            self.save_memory(messages)

            await self.on_task_end(messages)
            await self.cleanup_tools()
            yield messages
        except Exception as e:
            import traceback
            logger.warning(traceback.format_exc())
            if hasattr(self.config, 'help'):
                logger.error(
                    f'[{self.tag}] Runtime error, please follow the instructions:\n\n {self.config.help}'
                )
            raise e