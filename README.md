# ms-agent / AutoCoding
基于 ModelScope Agent 的多智能体自动编码工作流示例。

## 主要特性
- **配置驱动**：通过 `workflow.yaml` 与 `agent.yaml`/`agent.yml` 描述多智能体的有向流程与各自能力，无需改代码即可调整。
- **多模型兼容**：支持 OpenAI 兼容接口与 ModelScope Inference，按需切换 `llm.service`、`model`、`openai_base_url`、`modelscope_api_key`。
- **丰富工具链**：文件系统、Web 搜索/研究、RAG、Kaggle、Shell 执行、分任务拆解、状态切换等工具可按需启用。
- **可插拔记忆**：内置状态记忆与缓存，支持 `--load_cache` 复用历史对话/工具调用。
- **CLI 一键运行**：`python -m ms_agent.cli.cli run --config ...` 即可启动单 agent 或多 agent 工作流，可提供首次 query 或进入交互模式。

## 目录速览
```
ms_agent/             # 核心库（agent、workflow、tools、memory 等）
projects/deepcodingresearch/
	workflow.yaml       # 示例多智能体流程图
	research.yaml       # 研究 agent 配置
	coding.yaml         # 编码 agent 配置
	refine.yaml         # 测试/调试 agent 配置
requirements.txt      # 依赖列表
unit_test/            # 单元测试样例
```

## 环境准备
- Python 3.11（建议使用 conda/venv）
- 已获取的 OpenAI 兼容 API Key 或 ModelScope API Key

```bash
conda create -n autocoding python=3.11
conda activate autocoding
pip install -r requirements.txt
# 可在项目根目录创建 .env，写入 OPENAI_API_KEY / OPENAI_BASE_URL / MODELSCOPE_API_KEY 等
```

## 快速运行
以示例工作流 `projects/deepcodingresearch/workflow.yaml` 为例：

```bash
PYTHONPATH=. python ms_agent/cli/cli.py run --config projects/deepcodingresearch --query '查看目录下的instructions.txt文件并执行相关任务' --trust_remote_code true
```

- 若省略 `--query`，将进入交互模式。
- `--load_cache true`：复用上一次的对话/工具调用缓存。
- `--trust_remote_code true`：信任远端模型仓库中的自定义代码（从 ModelScope 拉取配置时需显式开启）。
- `--mcp_config` / `--mcp_server_file`：额外挂载 MCP server 配置。
- `--animation_mode auto|human`：视频生成场景的动画模式透传给下游。

## 配置说明
### 工作流 (`workflow.yaml`)
定义节点关系与各节点使用的 agent 配置文件：
```yaml
research:
    next: [coding]
    agent_config: research.yaml
coding:
    next: [structure_evaluate, research, refine]
    agent_config: coding.yaml
structure_evaluate:
    next: [coding]
    agent_config: structure_evaluate.yaml
refine:
    next: [exit, coding]
    agent_config: refine.yaml
exit:
    description: Exit the task when all the code works correctly.
```

### 单 Agent (`*.yaml`)
关键字段示例（摘自 `projects/deepcodingresearch/coding.yaml`）：
```yaml
llm:
	service: openai
	model: qwen3-max
	openai_api_key: <your_key>
	openai_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
agent: coding
type: state_llmagent
generation_config:
	temperature: 0.2
	max_tokens: 32000
prompt:
	system: |
		# 系统指令...
callbacks:
	- callbacks/tool_use_callback
tools:
	state_transition: { mcp: false }
	file_system:
		mcp: false
		ignore_files: [paper.md]
memory:
	- name: statememory
		user_id: "code_scratch"
max_chat_round: 100
tool_call_timeout: 30000
output_dir: new_output
```

> 小贴士：`Config.supported_config_names` 支持 `workflow.yaml|yml` 与 `agent.yaml|yml`，可在本地目录或 ModelScope 远端仓库之间切换（`--config <repo_id>`）。

## 常用工具能力
- `state_transition`：在工作流节点间传递状态/结果。
- `file_system`：读写/列目录等文件操作（可配置忽略/长度限制）。
- `document_inspector`：结构化解析文档内容。
- `deepresearch` / `web_search` / `web_research`：网页检索与研究。
- `split_task`：将编码任务拆解为可并行的子任务。
- `rag_tool`：基于 RAG 的检索与问答。
- `kaggle_tools`：下载/操作 Kaggle 数据集。
- `run_shell`：受控执行 Shell 命令。

## 存储与缓存
- `memory/`：状态记忆实现；配置中通过 `memory` 块启用。
- `new_output/`（默认可自定义）：模型输出与工具结果的落盘目录。
- `--load_cache true`：读取上次任务缓存，便于失败重试。

## 测试与开发
### 快速执行
```bash
python -m unit_test.test_deepresearch
python -m unit_test.test_rag
...
```
对于**test_state_memory**,可以通过前面的执行project的命令来执行，不过需要修改路径

### 单测覆盖点
- `test_deepresearch.py`：调用 `DeepresearchTool` 的 research 能力示例。
- `test_document_tool.py`：使用 `document_inspector` 对 `unit_test/paper.pdf` 做问答抽取（输出写入 `./unit_test`）。
- `test_rag.py`：初始化多路 `llama-index` RAG，构建索引后进行查询并打印调试信息。
- `test_file_parser.py`：演示 `SingleFileParser` 真实文件解析与缓存命中校验（需将示例中的文件路径与缓存目录替换为本地可用路径）。
- `test_state_memory/`：自定义拍卖工作流的状态迁移配置样例 (`workflow.yaml` 等)。

> 部分测试需要外部依赖：
> - LLM/RAG/视觉解析相关测试需有效的 OpenAI 兼容或 ModelScope API Key（放入 `.env` 或直接设环境变量）。
> - `test_file_parser.py` 需要你提供真实的本地文件路径，并在有图片时准备视觉解析所需的 Key。


### Examples

#### Kaggle 竞赛

在 `run_kaggle.sh` 中配置你的 API Key 和 Kaggle API TOKEN 后运行：

```bash
sh run_kaggle.sh
```

即可启动一个多智能体工作流，自动下载数据集、分析任务、生成代码并提交结果。

#### miniGPT
在 `run_gpt.sh` 中配置你的 API Key 后运行：

```bash
sh run_gpt.sh
```


