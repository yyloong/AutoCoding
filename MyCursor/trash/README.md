# MyCursor 多 Agent 协作工作流

这是一个基于 LoopWorkflow 的多 Agent 协作系统，用于自动化软件开发流程。

## 工作流架构

```
用户需求
   ↓
[Researcher] → analysis.md (需求调研报告)
   ↓
[Planner] → plan.md (开发计划)
   ↓
┌─────────────────────────────┐
│         Loop 循环            │
│                             │
│  [Coder] → 代码 + README.md │
│     ↓                       │
│  [Refiner] → report.md      │
│     ↓                       │
│  accept=true? ──No──┐       │
│     │               │       │
│    Yes           feedback.md│
│     ↓               │       │
│   退出 ←────────────┘       │
└─────────────────────────────┘
```

## Agent 职责

### 1. Researcher (调研专家)
- **输入**: 用户需求
- **职责**: 
  - 理解用户需求
  - 在互联网上搜索相关技术方案
  - 分析技术选型和实现方案
  - 评估项目规模
- **输出**: `analysis.md` (需求分析与技术方案)
- **工具**: 
  - `web_research`: 互联网搜索
  - `file_system`: 文件读写
  - `exit_task`: 退出任务

### 2. Planner (规划师)
- **输入**: `analysis.md` + 用户需求
- **职责**:
  - 读取调研报告
  - 将项目分解为具体的开发步骤
  - 制定详细的任务清单
- **输出**: `plan.md` (开发计划)
- **工具**:
  - `file_system`: 文件读写
  - `exit_task`: 退出任务

### 3. Coder (开发工程师)
- **输入**: `plan.md` + `feedback.md` (如果有)
- **职责**:
  - 按照开发计划逐步实现功能
  - 编写高质量、可运行的代码
  - 使用工具验证代码正确性
  - 生成项目文档
- **输出**: 
  - 项目代码文件
  - `README.md` (项目文档)
- **工具**:
  - `file_system`: 文件读写
  - `python_executor`: 执行 Python 代码片段 (待实现)
  - `code_executor`: 运行完整项目 (待实现)
  - `web_search`: 搜索技术文档
  - `exit_task`: 退出任务

### 4. Refiner (审查专家)
- **输入**: Coder 生成的代码 + `README.md`
- **职责**:
  - 按照 README 安装依赖
  - 运行项目并测试功能
  - 检查代码质量和功能完整性
  - 生成审查报告
- **输出**:
  - `report.md` (审查报告，包含 `accept` 字段)
  - `feedback.md` (如果 accept=false，给出修复建议)
- **工具**:
  - `file_system`: 文件读写
  - `environment_setup`: 安装依赖 (待实现)
  - `code_executor`: 运行项目 (待实现)
  - `shell_executor`: 执行 shell 命令 (待实现)
  - `exit_task`: 退出任务

## 循环机制

1. **Coder** 根据 `plan.md` 和 `feedback.md`（如果有）编写代码
2. **Refiner** 运行代码并审查，生成 `report.md`
3. 检查 `report.md` 中的 `accept` 字段：
   - 如果 `accept: true`，退出循环，项目完成
   - 如果 `accept: false`，Refiner 生成 `feedback.md`，返回步骤 1
4. 最多循环 10 次（可配置）

## 配置文件说明

### workflow.yaml
主工作流配置文件，定义了整个流程：

```yaml
type: LoopWorkflow

researcher:
  agent_config: researcher.yaml

planner:
  agent_config: planner.yaml
  
loop:
  coder:
    agent_config: coder.yaml
    breaker: false

  refiner:
    agent_config: refiner.yaml
    breaker: true
```

### 各 Agent 配置文件

每个 agent 的配置文件包含：
- `llm`: LLM 服务配置（模型、API 密钥等）
- `generation_config`: 生成参数（温度、top_k 等）
- `prompt.system`: 系统提示词（定义 agent 的职责和工作流程）
- `tools`: 可用的工具列表
- `code_file`: Agent 实现类（通常是 `simple_agent`）
- `max_chat_round`: 最大对话轮数
- `tool_call_timeout`: 工具调用超时时间
- `output_dir`: 输出目录

## 使用方法

### 1. 配置 API 密钥

在各个 yaml 文件中填入你的 API 密钥：

```yaml
llm:
  openai_api_key: your_api_key_here
```

### 2. 运行工作流

```bash
# 假设使用 ms_agent 框架运行
python -m ms_agent.workflow.run \
  --config MyCursor/workflow.yaml \
  --input "你的项目需求描述"
```

### 3. 查看输出

所有生成的文件将保存在 `output/` 目录下：
- `analysis.md`: 需求分析报告
- `plan.md`: 开发计划
- 项目代码文件
- `README.md`: 项目文档
- `report.md`: 审查报告
- `feedback.md`: 修复建议（如果需要）

## 待实现的工具

以下工具在配置文件中已定义，但需要实际实现：

1. **python_executor**: 执行 Python 代码片段
   - 用于 Coder 验证代码语法和逻辑
   - 输入：Python 代码字符串
   - 输出：执行结果或错误信息

2. **code_executor**: 运行完整项目
   - 用于 Coder 和 Refiner 测试整体功能
   - 输入：项目路径和运行命令
   - 输出：程序输出和退出码

3. **environment_setup**: 安装项目依赖
   - 用于 Refiner 设置测试环境
   - 支持 pip、npm、uv 等包管理器
   - 输入：依赖文件路径（requirements.txt、package.json 等）
   - 输出：安装结果

4. **shell_executor**: 执行 shell 命令
   - 用于 Refiner 执行各种测试命令
   - 输入：shell 命令字符串
   - 输出：命令输出和退出码

## 自定义和扩展

### 修改提示词

编辑各个 agent 的 yaml 文件中的 `prompt.system` 部分，可以调整 agent 的行为。

### 调整循环次数

在运行时传入参数：

```bash
python -m ms_agent.workflow.run \
  --config MyCursor/workflow.yaml \
  --input "你的需求" \
  --max_iterations 5
```

### 添加新的 Agent

1. 创建新的 yaml 配置文件
2. 在 `workflow.yaml` 中添加新的 agent 配置
3. 如果需要在循环中，添加到 `loop` 部分

### 修改 LoopWorkflow 逻辑

编辑 `loop_workflow.py` 文件，可以自定义：
- 文件注入逻辑（`_inject_plan_and_feedback` 方法）
- 循环退出条件（`_should_break_loop` 方法）
- 任务执行顺序

## 注意事项

1. **API 配额**: 整个流程可能会消耗大量 API 调用，注意控制成本
2. **输出目录**: 确保 `output_dir` 有写入权限
3. **超时设置**: 对于大型项目，可能需要增加 `tool_call_timeout`
4. **模型选择**: 
   - Researcher/Planner 使用 `qwen3-coder-flash`（速度快）
   - Coder/Refiner 使用 `qwen3-coder-plus`（质量高）
5. **循环控制**: 如果循环次数过多，检查 Refiner 的审查标准是否过于严格

## 示例

### 示例需求

```
请帮我开发一个简单的待办事项管理工具，要求：
1. 使用 Python 实现
2. 支持添加、删除、查看待办事项
3. 数据保存到 JSON 文件
4. 提供命令行界面
```

### 预期输出

1. `analysis.md`: 分析 Python CLI 开发方案，推荐使用 argparse 和 json 模块
2. `plan.md`: 分解为 5 个步骤（数据模型、文件操作、CLI 接口、主程序、文档）
3. 代码文件: `todo.py`（主程序）、`requirements.txt`（依赖）
4. `README.md`: 详细的使用说明
5. `report.md`: 审查报告，确认功能正常

## 许可证

Copyright (c) Alibaba, Inc. and its affiliates.

