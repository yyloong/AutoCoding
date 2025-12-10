# Copyright (c) Alibaba, Inc. and its affiliates.
import tiktoken
import asyncio
from ms_agent.llm.llm import LLM
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ms_agent.llm.utils import Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig

from .base import Memory
import os

logger = get_logger()

select_message_prompt = """
    You are an AI assistant managing the state memory for an agent.Here is a chat history
    among different agents:<{messages}>, agents will not see the messages after he sending his last
    message so you need to select some important messages to tell them.
    Now it is time for <{agent_name}> to work,please output information that you think is the most important for him.
    If the message is very important, please try to keep all the information,such as some url,path,useful analysis,concrete code snippets or instructions.
    You should show the source of the information in your answer,such as which agent sent it.
    You should try to keep the context for some continue tasks,such as A ask B to do something for him,
    and B ask C to help him, then you should keep A's request when you select messages.
    Think carefully before you make your choice since your answer will directly affect whether the agent can get the necessary information.
    You should output your answer in the following format:
    <start>
    agent_name request to contact you
    agent_name1 message: important_information
    agent_name2 message: important_information
    ...
    <end>
"""

add_message_prompt = """
You are an AI assistant managing the state memory for an agent.Here is a chat history
among different agents:<{messages}>,now <{agent_name}> has sent a new message:<{new_message}>,
please output information that you think is important for other agents to know.
If the message is very important, please try to keep all the information,such as some url,path,useful analysis,concrete code snippets or instructions.
For some less important messages,you can make some summary.
Such as some important details or instructions.
You should output your answer in the following format:
<start> 
important_information
<end>
"""

condense_prompt = """
You are an AI assistant managing the state memory for an agent.You are given a message or 
a list of messages coming from the working process from an agent,it/they can be agent's thought,plan,action or 
message from the tool that is called by the agent.Here is/are the message:<{message}>,please 
condense the message(s) to keep the important information.Please make sure the information removing 
won't influence the agent's future decision making a lot.And you should express in user's point of view.
For example,you should use "You have..."
And you should output your answer in the following format:
<start>
condensed_message
<end>
```
"""


class StateMemoryManager:
    """Manager for exact state memory instances (Blackboard pattern)."""

    _instances: Dict[str, "ExactStateMemory"] = {}
    shared_messages_list: List[str] = []
    shared_memory_path = os.path.join("memory", "state_memory", "overall_shared_memory.json")

    @classmethod
    def get_memory(cls, config: DictConfig) -> "ExactStateMemory":
        """Get or create a state memory instance based on configuration."""
        # 使用 session_id 隔离不同任务的执行上下文
        user_id = getattr(config, "user_id", "")
        assert user_id, "user_id must be provided in config for StateMemoryManager"

        if user_id not in cls._instances:
            logger.info(f"Creating new exact state memory for user: {user_id}")
            cls._instances[user_id] = ExactStateMemory(config)
        else:
            logger.debug(f"Reusing existing state memory for user: {user_id}")

        return cls._instances[user_id]

    @classmethod
    def clear_memory(cls, config: DictConfig = None):
        """Clear memory instances."""
        if config is None:
            cls._instances.clear()
            logger.info("Cleared all state memory instances")
        else:
            session_id = getattr(config, "session_id", "default_session")
            if session_id in cls._instances:
                del cls._instances[session_id]
                logger.info(f"Cleared state memory for session: {session_id}")


class ExactStateMemory(Memory):
    """
    Exact State Memory implementation for Agents.
    Instead of vector retrieval, this maintains a deterministic structure of:
    1. Global Artifacts (The "Blackboard" - variables accessible to all states)
    2. Execution Trace (Chronological history of state transitions)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        
    async def set_base_config(self, config: DictConfig):
        """Allow updating config at runtime."""
        self.config = config
        self.tag = getattr(config, "tag", None)
        assert (
            self.tag is not None
        ), "Tag must be provided in config for ExactStateMemory"
        self.tag = self.tag.split("_")[0]
        memory_file = self.tag + "_state_memory.json"
        self.messages_save_path = getattr(
            config.memory,
            "messages_save_path",
            os.path.join(os.path.dirname(StateMemoryManager.shared_memory_path), memory_file),
        )
        self.condense_limit = getattr(
            config.memory, "condense_limit", 200000
        )  # if tokens exceed, condense the memory
        self.single_msg_limit = getattr(config.memory, "single_msg_limit", 30000)
        self.last_n_messages = getattr(config.memory, "last_n_messages", 8)
        self.memory_type = getattr(config.memory, "memory_type", "long-term")
        self.initialize_condense_llm()

    def initialize_condense_llm(self):
        self.llm = LLM.from_config(self.config)

    def extract_answer_from_response(self, str_response: str) -> str:
        """Extract the main answer from LLM response formatted with triple backticks."""
        if "<start>" in str_response and "<end>" in str_response:
            parts = str_response.split("<start>")
            parts = parts[1].split("<end>")
            return parts[0].strip()
        return None 
    
    async def condense_str(self, content: str) -> str:
        """Condense the state memory when it exceeds token limits."""
        # Placeholder for condensing logic
        prompt = condense_prompt.format(message=content)
        answer = None
        while answer is None:
            logger.info("Condensing state memory content...")
            for stream in self.llm.generate([Message(role="user", content=prompt)]):
                response = stream.content
            answer = self.extract_answer_from_response(response)
        return answer

    async def run(self, messages: List[Message]) -> List[Message]:
        """
        Injects the current Global Artifacts and Recent State History into the system prompt.
        This ensures the Agent acts on precise, current information without retrieval loss.
        """
        # get tokens count
        token_consume = 0
        if messages[-1].role != "assistant":
            return messages

        token_list = self.count_tokens(messages)
        token_consume = sum(token_list)
        logger.info(f"StateMemoryManager token_consume: {token_consume}")
        if token_consume < self.condense_limit:
            return messages

        assert messages[0].role == "system", "First message must be system message"
        if len(messages) > 1:
            assert messages[1].role == "user", "Second message must be user message"

        if len(messages) <= self.last_n_messages + 2:
            for i in range(
                2, len(messages) - 2
            ):  # skip the first two and last two messages
                if token_list[i] > self.single_msg_limit:
                    is_tool = (
                        messages[i].role == "assistant" and 
                        (getattr(messages[i], "tool_calls", None) or getattr(messages[i], "function_call", None))
                    )
                    if is_tool:
                        logger.info(f"Skipping compression for message at index {i} (Tool Call detected)")
                        continue

                    logger.info(
                        f"Condensing message at index {i} with token count {token_list[i]}"
                    )
                    messages[i].content = await self.condense_str(messages[i].content)
            return messages
        else:
            logger.info("Executing Rolling Summary logic...")

            head_msgs = messages[:2]
            # ---------------------------------------------------------
            # [新增逻辑 2] 动态调整切分点 (保护 Rolling Summary 的边界)
            # ---------------------------------------------------------
            # 默认保留最近 N 条
            tail_keep_count = self.last_n_messages
            
            # 计算原本计划放入 middle 的最后一条消息的索引
            # messages 总长度 - 尾部保留长度 - 1 = middle 部分的最后一个索引
            last_middle_index = len(messages) - tail_keep_count - 1

            # 确保索引没有越界侵入 head 部分 (head 占用了 index 0 和 1)
            if last_middle_index >= 2:
                candidate_msg = messages[last_middle_index]
                
                # 判断是否为 assistant 的工具调用
                # 兼容 tool_calls (新) 和 function_call (旧)
                is_tool_call = (
                    candidate_msg.role == "assistant" and 
                    (getattr(candidate_msg, "tool_calls", None) or getattr(candidate_msg, "function_call", None))
                )

                if is_tool_call:
                    logger.info("Detected Tool Call at the end of compression range. Expanding tail to preserve it.")
                    # 将尾部保留数量 +1，这样这条消息就会被划分到 recent_msgs 中，而不是 middle_msgs
                    tail_keep_count += 1
            # ---------------------------------------------------------

            # Tail: 根据调整后的计数保留最近消息
            recent_msgs = messages[-tail_keep_count :]

            # Middle: 待压缩区域
            middle_msgs = messages[2 : -tail_keep_count]
            
            if not middle_msgs:
                return messages
                
            # 2. 压缩 Middle 部分
            message_str = "\n".join([msg.content for msg in middle_msgs])
            message_str = await self.condense_str(message_str)
            
            # 构造压缩后的消息（通常作为 User 角色或 System Note 插入）
            middle_msgs = [Message(role="user", content=message_str)]
            
            messages = head_msgs + middle_msgs + recent_msgs
            logger.info("Completed Rolling Summary.")
            return messages

    async def load_messages(self,messages: List[Message]=None) -> Optional[List[Message]]:
        if os.path.exists(StateMemoryManager.shared_memory_path):
            with open(StateMemoryManager.shared_memory_path, "r") as f:
                shared_messages = json.load(f)
                StateMemoryManager.shared_messages_list = [
                    msg for msg in shared_messages
                ]
        if len(StateMemoryManager.shared_messages_list) == 0:
            return None
        if len(messages) > 2:
            logger.info("Messages already exist, skipping load from state memory.")
            return None
        if not os.path.exists(self.messages_save_path):
            logger.info(f"State memory file {self.messages_save_path} does not exist,Taking it as a new state.")
            os.makedirs(os.path.dirname(self.messages_save_path), exist_ok=True)
            select_message = await self.select_messages_for_agent(self.tag)
            messages.append(Message(role='user', content=select_message))
            return None

        with open(self.messages_save_path, "r") as f:
            messages = json.load(f)
            messages = [Message(**msg) for msg in messages]
        select_message = await self.select_messages_for_agent(self.tag)
        select_message = "state transition back ,with important information:\n" + select_message
        message = Message(role='user', content=select_message)
        messages.append(message)
        return messages

    async def add_message_to_shared(self, message: str):
        """Add a message to the shared messages list."""
        if message is None or message.strip() == "":
            return
        prompt = add_message_prompt.format(
            messages="\n".join([msg for msg in StateMemoryManager.shared_messages_list]),
            agent_name=self.tag,
            new_message=message,
        )
        answer = None
        while answer is None:
            logger.info(f"Adding message to shared memory for agent {self.tag}")
            response = ""
            for stream in self.llm.generate([Message(role="user", content=prompt)]):
                response = stream.content
            answer = self.extract_answer_from_response(response)
        answer = f"agent: {self.tag} message: {answer}"
        StateMemoryManager.shared_messages_list.append(answer)
    
    async def select_messages_for_agent(self, agent_name: str) -> str:
        """Select important messages for a specific agent."""
        prompt = select_message_prompt.format(
            messages="\n".join([msg for msg in StateMemoryManager.shared_messages_list]),
            agent_name=agent_name,
        )
        answer = None
        while answer is None:
            logger.info(f"Selecting messages for agent {agent_name}")
            for stream in self.llm.generate([Message(role="user", content=prompt)]):
                response = stream.content
            answer = self.extract_answer_from_response(response)
        return answer
    
    async def save_messages(self, messages: List[Message]):
        os.makedirs(os.path.dirname(self.messages_save_path), exist_ok=True)
        with open(self.messages_save_path, "w") as f:
            json.dump(
                [msg.to_dict() for msg in messages],
                f,
                indent=4,
                ensure_ascii=False,
            )
        logger.info(f"Saved messages to {self.messages_save_path}")
        with open(StateMemoryManager.shared_memory_path, "w") as f:
            json.dump(
                [msg for msg in StateMemoryManager.shared_messages_list],
                f,
                indent=4,
                ensure_ascii=False,
            )
        logger.info(f"Saved shared messages to {StateMemoryManager.shared_memory_path}")

    def count_tokens(self, messages: List[Message]) -> List[int]:
        """
        使用 tiktoken 对每条消息进行精确 Token 计数。
        适用于 DeepSeek 和 Qwen (使用 cl100k_base 编码)。

        Args:
            messages: 消息列表
            known_total_tokens: (在此新逻辑中不再需要，仅保留参数以兼容接口)
        """
        if not messages:
            return []

        # Qwen 和现代高性能模型通常使用 cl100k_base
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # Fallback 到 gpt2 (r50k_base) 以防环境问题，虽然不太可能
            logger.warning(f"Could not load cl100k_base, falling back to gpt2: {e}")
            encoding = tiktoken.get_encoding("gpt2")

        tokens_list = []

        for msg in messages:
            # 1. 基础消耗：每条消息的固有开销
            # OpenAI/Qwen 格式通常为: <|im_start|>role\nContent<|im_end|>\n
            # 这大约是 3-4 个 tokens。这里按常用的 3 tokens 估算。
            num_tokens = 3 
            
            # 2. 计算 Content 的 Token
            content = msg.content
            if content:
                # 确保 content 是字符串
                num_tokens += len(encoding.encode(str(content)))
            
            # 3. (可选) 计算 Tool Calls 的 Token
            # 如果你的 Message 对象包含 tool_calls 且不为空，也需要统计
            # 这里假设 msg 对象可能有 tool_calls 属性
            tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "function_call", None)
            if tool_calls:
                # 简单估算：将工具调用的 JSON 结构转为字符串进行编码
                import json
                # 处理 tool_calls 是对象还是字典的情况
                try:
                    if isinstance(tool_calls, list):
                        dump_str = json.dumps([t if isinstance(t, dict) else t.model_dump() for t in tool_calls])
                    else:
                        dump_str = str(tool_calls)
                    num_tokens += len(encoding.encode(dump_str))
                except Exception:
                    # 如果转换失败，保守增加一个固定值
                    num_tokens += 10

            # 赋值给 message 对象 (保持原逻辑副作用)
            # 注意：如果 msg 是 Pydantic 模型，可能需要 setattr 或 msg.prompt_tokens = ...
            try:
                msg.prompt_tokens = num_tokens
            except AttributeError:
                pass # 如果 msg 对象不支持设置属性则跳过

            tokens_list.append(num_tokens)

        # 4. 加上回复前缀的 Token (Reply Primer)
        # 通常是 <|im_start|>assistant<|im_sep|>，约为 3 tokens
        # 在计算 prompt cache 时这部分是针对整个 prompt 的，但如果分配到单条消息，可以忽略或加在最后一条
        
        return tokens_list