# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

import json
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class CodingCallback(Callback):
    """Add more prompts when coding
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_tool_call(self, runtime: Runtime, messages: List[Message]):
        # tool name is not 'split_to_sub_task', ut is 'SplitTask---split_to_sub_task'
        if not messages[-1].tool_calls or 'split_to_sub_task' not in messages[
                -1].tool_calls[0]['tool_name']:
            return
        assert messages[0].role == 'system'
        arguments = messages[-1].tool_calls[0]['arguments']
        arguments = json.loads(arguments)
        tasks = arguments['tasks']
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        for task in tasks:
            task['_system'] = task['system']
            task['system'] = f"""{task["system"]}

The PRD of this project:

{messages[2].content}

Strictly follow the steps:
1.Before writing each file, analyze its dependencies. List the modules or files it imports/requires and read their implementations first to ensure compatibility.
For example:
```
// My file, `processor.py`, imports `User` from `models.py` and `connect_db` from `database.py`. I should read them first to understand the data structure and function signatures.
```

2.Read the target code file itself (if it already exists) to understand its current logic and prevent making breaking changes to existing functionality.
You may read several files in step 1 and step 2; this is good practice to understand the project. You may also read other files if necessary, like configuration files (config.json) or dependency manifests (package.json, pyproject.toml), to enhance your understanding.

3.write your code using the provided tools:

The header (filepath) will be used for saving the file. Therefore, you must generate it strictly in this format.

4.Ensure robust error handling and logging. Do not let your code crash silently. Use standard logging practices for the language (e.g., console.log/console.error in JavaScript/Node.js, print() or the logging module in Python) to output meaningful status and error messages to the terminal.

5.Here are extra instructions for writing high-quality code:
Code Clarity and Readability: Use descriptive and unambiguous variable and function names. Write clean, well-structured code that is easy for other developers to understand and maintain.
Modularity and Structure: Decompose complex problems into smaller, reusable functions or classes. Adhere to the Single Responsibility Principle, where each module or function does one thing well.
Language-Specific Best Practices: Write idiomatic code that follows the common conventions and style guides of the language (e.g., write "Pythonic" code following PEP 8 in Python; follow common asynchronous patterns in Node.js).
Comments and Documentation: Add comments to explain complex, non-obvious, or important parts of your code. For public-facing functions, classes, or modules, provide clear documentation strings (docstrings in Python, JSDoc in JavaScript).
Resource Handling: If the code needs to access external resources (like configuration files, data files, or API endpoints), do not hardcode paths or URLs directly in the logic. Pass them as arguments, or load them from a dedicated configuration module.
Output Format: Your output must only be text-based source code files. Do not attempt to generate or output binary files (like images, database files, or compiled artifacts).

6.Pay attention to the correctness of the path you provide in the filename. 

7.strictly follow the path you are required to writing files

8.you have no right to run the code and do not need to consider whether the code can run successfully or not.

9.Only consider the file you are required to write, do not consider other files,pay attention!

10.If you find the file you are going to read of write does not exist,first read the files.json to check the path whether it is right .

11.If you ensure you have finished all the tasks(pay attention that it means you create the required files by using tools rather than given in your output) assigned to you,use tools to exit your task.

Now,begin:

""" # noqa
        messages[-1].tool_calls[0]['arguments'] = json.dumps({'tasks': tasks})

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        if not messages[-2].tool_calls or 'split_to_sub_task' not in messages[
                -2].tool_calls[0]['tool_name']:
            return
        assert messages[0].role == 'system'
        arguments = messages[-2].tool_calls[0]['arguments']
        arguments = json.loads(arguments)
        tasks = arguments['tasks']
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        for task in tasks:
            if '_system' in task:
                task['system'] = task['_system']
                task.pop('_system')
        messages[-2].tool_calls[0]['arguments'] = json.dumps({'tasks': tasks})
