# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import inspect
import os.path
import sys
import uuid
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import json
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback, callbacks_mapping
from ms_agent.llm.llm import LLM
from ms_agent.llm.utils import Message
from ms_agent.memory import Memory, memory_mapping
from ms_agent.memory.mem0ai import Mem0Memory, SharedMemoryManager
from ms_agent.memory.statememory import StateMemoryManager, ExactStateMemory
from ms_agent.rag.base import RAG
from ms_agent.tools import ToolManager
from ms_agent.agent.llm_agent import LLMAgent
from ms_agent.utils import async_retry, read_history, save_history
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR, DEFAULT_TAG, DEFAULT_USER
from ms_agent.utils.logger import logger
from omegaconf import DictConfig, OmegaConf

from ..config.config import Config, ConfigLifecycleHandler
from .base import Agent


class State_LLMAgent(LLMAgent):
    AGENT_NAME = "State_LLMAgent"

    def __init__(
        self,
        config: DictConfig = DictConfig({}),
        tag: str = DEFAULT_TAG,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        if not hasattr(config, "llm"):
            default_yaml = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "agent.yaml"
            )
            llm_config = Config.from_task(default_yaml)
            config = OmegaConf.merge(llm_config, config)
        super().__init__(config, tag, trust_remote_code,**kwargs)
        self.config.tag = self.tag
        self.config["next_tasks"] = kwargs.get("next_tasks", [])
        self.config["tasks_descriptions"] = kwargs.get("tasks_descriptions_map", {})
        self.next_task = None
        self.messages = None
        self.exit_description = None

    async def load_memory(self):
        """Initialize and append memory tool instances based on the configuration provided in the global config.

        Raises:
            AssertionError: If a specified memory type in the config does not exist in memory_mapping.
        """
        self.config: DictConfig
        if hasattr(self.config, 'memory'):
            for _memory in (self.config.memory or []):
                memory_type = getattr(_memory, 'name', 'default_memory')
                assert memory_type in memory_mapping, (
                    f'{memory_type} not in memory_mapping, '
                    f'which supports: {list(memory_mapping.keys())}')

                # Use LLM config if no special configuration is specified
                llm_config = getattr(_memory, 'llm', None)
                if llm_config is None:
                    service = self.config.llm.service
                    config_dict = {
                        'model':
                        _memory.summary_model if hasattr(
                            _memory, 'summary_model') else getattr(
                                self.config.llm, 'model', None),
                        'provider':
                        'openai',
                        'openai_base_url':
                        getattr(self.config.llm, f'{service}_base_url', None),
                        'openai_api_key':
                        getattr(self.config.llm, f'{service}_api_key', None),
                        'max_tokens':
                        getattr(_memory, 'max_tokens', 4096),
                    }
                    llm_config_obj = OmegaConf.create(config_dict)
                    setattr(_memory, 'llm', llm_config_obj)
                if memory_type == 'mem0':
                    shared_memory = SharedMemoryManager.get_shared_memory(
                        _memory)
                    self.memory_tools.append(shared_memory)
                else:
                    self.memory_tools.append(
                        memory_mapping[memory_type](_memory))

                for memory in self.memory_tools:
                    # In case any memory tool need other information
                    await memory.set_base_config(self.config)


    @async_retry(max_attempts=Agent.retry_count, delay=1.0)
    async def step(
        self, messages: List[Message]
    ) -> AsyncGenerator[List[Message], Any]:  # type: ignore
        
        print(messages)
        messages = deepcopy(messages)
        if (not self.load_cache) or messages[-1].role != "assistant":
            await self.on_generate_response(messages)
            tools = await self.tool_manager.get_tools()

            if self.stream:
                self.log_output("[assistant]:")
                _content = ""
                is_first = True
                _response_message = None
                for _response_message in self.llm.generate(messages, tools=tools):
                    if is_first:
                        messages.append(_response_message)
                        is_first = False
                    new_content = _response_message.content[len(_content) :]
                    sys.stdout.write(new_content)
                    sys.stdout.flush()
                    _content = _response_message.content
                    messages[-1] = _response_message
                    yield messages
                sys.stdout.write("\n")
            else:
                _response_message = self.llm.generate(messages, tools=tools)
                if _response_message.content:
                    self.log_output("[assistant]:")
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
        messages = await self.condense_memory(messages)

        if _response_message.tool_calls:
            messages = await self.parallel_tool_call(messages)

        if (
            _response_message.tool_calls
            and _response_message.tool_calls[-1]["tool_name"] == "finish---exit_task"
        ):
            self.runtime.should_stop = True

        if (
            _response_message.tool_calls
            and _response_message.tool_calls[-1]["tool_name"].startswith(
                "state_transition---"
            )
            and messages[-1].content == "Successful"
        ):
            self.runtime.should_stop = True
            self.next_task = _response_message.tool_calls[-1]["tool_name"].split("---")[-1]
            exit_message = _response_message.tool_calls[-1]["arguments"]
            exit_message = json.loads(exit_message)
            self.exit_description = f"request to make a state transition to {self.next_task} with message"+exit_message.get("message", "")
            self.messages = messages

        await self.after_tool_call(messages)
        self.log_output(
            f"[usage] prompt_tokens: {_response_message.prompt_tokens}, "
            f"completion_tokens: {_response_message.completion_tokens}"
        )
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
            
            self.runtime.tag = self.tag

            if messages is None:
                messages = self.query
            
            self.config, self.runtime, messages = self.read_history(messages)

            if self.runtime.round == 0:
                # 0 means no history
                messages = await self.create_messages(messages)
                await self.on_task_begin(messages)

            for message in messages:
                if message.role != 'system':
                    self.log_output('[' + message.role + ']:')
                    self.log_output(message.content)

            if not self.runtime.should_stop:
                for memory_tool in self.memory_tools:
                    if isinstance(memory_tool, ExactStateMemory):
                        result = await memory_tool.load_messages(messages)
                        if result:
                            messages = result
                self.messages = messages

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
    
    async def exit_state(self):
        for memory_tool in self.memory_tools:
            if isinstance(memory_tool, ExactStateMemory):
                if self.exit_description:
                    await memory_tool.add_message_to_shared(self.exit_description)
                if self.messages:
                    await memory_tool.save_messages(self.messages)

    def next_flow(self):
        if self.next_task is None:
            logger.warning("Next task is None, try to debug it.")
            import pdb
            pdb.set_trace()
        return self.next_task

    def save_history(self, messages: List[Message], **kwargs):
        """
        Save current chat history to disk for future resuming.

        Args:
            messages (List[Message]): Current message history to save.
        """
        query = None
        if len(messages) > 1 and messages[1].role == 'user':
            query = messages[1].content
        elif messages:
            query = messages[0].content
        if not query:
            return

        if not getattr(self.config, 'save_history', True):
            return

        config: DictConfig = deepcopy(self.config)
        config.runtime = self.runtime.to_dict()
        config.next_task = self.next_task 
        ################33
        #save_history(
        #    self.output_dir, task=self.tag, config=config, messages=messages)
        ##################
        save_history(
            "./", task=self.tag, config=config, messages=messages
        )
    
    def read_history(self, messages: List[Message],
                     **kwargs) -> Tuple[DictConfig, Runtime, List[Message]]:
        """
        Load previous chat history from disk if available.(断点重连作用)

        Args:
            messages (List[Message]): Input message or history to resume from.

        Returns:
            Tuple[DictConfig, Runtime, List[Message]]: Updated config, runtime, and message history.
        """
        if isinstance(messages, str):
            query = messages
        else:
            query = messages[1].content
        if not query or not self.load_cache:
            return self.config, self.runtime, messages

        #config, _messages = read_history(self.output_dir, self.tag)
        config, _messages = read_history("./", self.tag)
        if config is not None and _messages is not None:
            if hasattr(config, 'runtime'):
                runtime = Runtime(llm=self.llm)
                runtime.from_dict(config.runtime)
                delattr(config, 'runtime')
            else:
                runtime = self.runtime
            if hasattr(config, 'next_task'):
                self.next_task = config.next_task
                delattr(config, 'next_task')
            if _messages[-1].role == 'tool':
                # Ignore and redo the last tool response
                # This is because it's the last calling, the unhandled error may be started from here
                _messages = _messages[:-1]
            return config, runtime, _messages
        else:
            return self.config, self.runtime, messages