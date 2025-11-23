# MyCursor 多 Agent 协作工作流

这是一个基于 LoopWorkflow 的多 Agent 协作系统，用于自动化软件开发流程。

## 项目结构

```
MyCursor/
├── workflow.yaml           # 主工作流配置
├── researcher.yaml         # Researcher Agent 配置
├── planner.yaml           # Planner Agent 配置
├── coder.yaml             # Coder Agent 配置
├── refiner.yaml           # Refiner Agent 配置
├── simple_agent.py        # Agent 基础实现
└── README.md              # 本文档
```

## 工作流架构

```
用户需求
   ↓
[Researcher] → analysis.md (需求调研报告)
   ↓
[Planner] → plan.md (开发计划)
   ↓
┌──────────────────────────────────────────────┐
│              Loop 循环（最多10次）              │
│                                              │
│  第1轮：                                      │
│  [Coder] 读取 plan.md                        │
│     ↓ (首次开发模式)                          │
│  生成代码 + README.md                         │
│     ↓                                        │
│  [Refiner] 测试并审查                         │
│     ↓                                        │
│  生成 report.md (含 accept 字段)              │
│     ↓                                        │
│  accept=true? ──Yes→ 退出循环                 │
│     │                                        │
│     No (第2轮开始)                            │
│     ↓                                        │
│  [Coder] 读取 README.md + report.md          │
│     ↓ (迭代修改模式)                          │
│  修复问题，更新代码                            │
│     ↓                                        │
│  [Refiner] 重新测试                           │
│     ↓                                        │
│  更新 report.md                               │
│     ↓                                        │
│  accept=true? ──Yes→ 退出循环                 │
│     │                                        │
│     No → 继续下一轮...                        │
└──────────────────────────────────────────────┘
```

## Agent 职责

### 1. Researcher (调研专家)
- **配置文件**: `MyCursor/researcher.yaml`
- **输入**: 用户需求
- **职责**: 
  - 理解用户需求
  - 在互联网上搜索相关技术方案
  - 分析技术选型和实现方案
  - 评估项目规模
- **输出**: `output/analysis.md` (需求分析与技术方案)
- **工具**: 
  - `web_research`: 互联网搜索
  - `file_system`: 文件读写
  - `exit_task`: 退出任务

### 2. Planner (规划师)
- **配置文件**: `MyCursor/planner.yaml`
- **输入**: `output/analysis.md` + 用户需求
- **职责**:
  - 读取调研报告
  - 将项目分解为具体的开发步骤
  - 制定详细的任务清单
- **输出**: `output/plan.md` (开发计划)
- **工具**:
  - `file_system`: 文件读写
  - `exit_task`: 退出任务

### 3. Coder (开发工程师)
- **配置文件**: `MyCursor/coder.yaml`
- **输入**: 
  - **首次开发**: `plan.md`（由 `ms_agent/workflow/loop_workflow.py` 动态注入）
  - **迭代修改**: `README.md` + `report.md`（由 `ms_agent/workflow/loop_workflow.py` 动态注入）
- **工作模式**: 
  - **首次开发模式**: 根据 plan.md 从零开始实现项目
  - **迭代修改模式**: 根据 report.md 的反馈修改现有代码
- **职责**:
  - 按照开发计划逐步实现功能（首次）或修复问题（迭代）
  - 编写高质量、可运行的代码
  - 使用工具验证代码正确性
  - 生成或更新项目文档
- **输出**: 
  - 项目代码文件（保存在 `output/` 目录）
  - `README.md` (项目文档)
- **工具**:
  - `file_system`: 文件读写
  - `run_code`: 运行代码文件进行验证
  - `web_search`: 搜索技术文档
  - `environment_set_up`: 安装依赖
  - `exit_task`: 退出任务
- **特殊机制**: 
  - 提示词由 `loop_workflow.py` 根据是否存在 `report.md` 动态生成
  - 首次开发时注入详细的开发流程指导
  - 迭代修改时注入问题修复流程指导

### 4. Refiner (审查专家)
- **配置文件**: `MyCursor/refiner.yaml`
- **输入**: Coder 生成的代码 + `output/README.md`
- **职责**:
  - 按照 README 安装依赖
  - 运行项目并测试功能
  - 检查代码质量和功能完整性
  - 生成审查报告
- **输出**:
  - `output/report.md` (审查报告，包含 `accept` 字段)
  - **注意**: 不再生成单独的 feedback.md，所有反馈都在 report.md 中
- **工具**:
  - `file_system`: 文件读写
  - `environment_set_up`: 安装依赖
  - `run_code`: 运行代码文件
  - `exit_task`: 退出任务

## 循环机制

### 工作流程

**第 1 轮（首次开发）**:
1. **Coder** 处于"首次开发模式"
   - 读取 `output/plan.md`（由 `loop_workflow.py` 自动注入）
   - 根据开发计划从零开始实现项目
   - 生成代码文件和 `output/README.md`
2. **Refiner** 运行代码并审查
   - 按照 README.md 的说明测试项目
   - 生成 `output/report.md`，包含 `accept` 字段
3. 检查 `report.md` 中的 `accept` 字段：
   - 如果 `accept: true` → 退出循环，项目完成 ✅
   - 如果 `accept: false` → 进入第 2 轮

**第 2 轮及后续（迭代修改）**:
1. **Coder** 处于"迭代修改模式"
   - 读取 `output/README.md` + `output/report.md`（由 `loop_workflow.py` 自动注入）
   - **不再读取** `plan.md`，专注于修复 report.md 中指出的问题
   - 按优先级修复：致命错误 → 功能缺陷 → 质量问题 → 文档问题
   - 更新代码和 README.md
2. **Refiner** 重新测试
   - 验证问题是否已修复
   - 更新 `output/report.md`
3. 检查 `accept` 字段，决定是否继续循环

### 关键特性

- **动态提示词**: `loop_workflow.py` 根据是否存在 `report.md` 动态生成不同的提示词
- **上下文切换**: 首次开发关注完整实现，迭代修改关注问题修复
- **最大循环次数**: 默认 10 次（可配置），避免无限循环
- **自动清理**: 每次运行前自动清理 `output/` 和 `memory/` 目录

### 循环退出条件

- ✅ `report.md` 中 `accept: true`
- ⚠️ 达到最大迭代次数（默认 10 次）

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

每个 agent 的配置文件（`MyCursor/*.yaml`）包含：
- `llm`: LLM 服务配置（模型、API 密钥等）
- `generation_config`: 生成参数（温度、top_k 等）
- `prompt.system`: 系统提示词
  - **Researcher/Planner/Refiner**: 静态提示词，定义在各自的 yaml 文件中
  - **Coder**: 简化的基础提示词，详细的工作流程由 `ms_agent/workflow/loop_workflow.py` 动态注入
- `tools`: 可用的工具列表
- `code_file`: Agent 实现类（使用 `simple_agent`，位于 `MyCursor/simple_agent.py`）
- `max_chat_round`: 最大对话轮数
- `tool_call_timeout`: 工具调用超时时间（毫秒）
- `output_dir`: 输出目录（默认为 `output`）

### 核心实现文件

- **`ms_agent/workflow/loop_workflow.py`**: 
  - LoopWorkflow 的核心实现
  - 负责协调各个 agent 的执行顺序
  - 动态生成 Coder 的提示词（`_get_initial_system_prompt()` 和 `_get_iteration_system_prompt()`）
  - 注入上下文信息（`_inject_plan_and_feedback()`）
  - 判断循环退出条件（`_should_break_loop()`）

- **`MyCursor/simple_agent.py`**:
  - Agent 的基础实现类
  - 继承自 `ms_agent.LLMAgent`
  - 处理消息流、工具调用、任务退出等逻辑

## 使用方法

### 1. 配置 API 密钥

设置环境变量

```bash
export OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}
export SERPER_KEY_ID=${YOUR_SERPER_KEY_ID}
export JINA_API_KEYS=${YOUR_JINA_API_KEYS}
```

### 2. 运行工作流

```bash
# 使用 ms_agent CLI 运行
python -m ms_agent.cli.cli run \
  --config MyCursor \
  --trust_remote_code true \
  --openai_api_key ${OPENAI_API_KEY} \
  --query "你的项目需求描述"

# 或者使用提供的脚本（需要先配置环境变量）
cd MyCursor
bash runs.sh
```

### 添加新的 Agent

1. 创建新的 yaml 配置文件
2. 在 `workflow.yaml` 中添加新的 agent 配置
3. 如果需要在循环中，添加到 `loop` 部分

### 修改 LoopWorkflow 逻辑

编辑 `ms_agent/workflow/loop_workflow.py` 文件，可以自定义：

- **`_inject_plan_and_feedback()`**: 控制注入给 Coder 的上下文
  - 首次开发：注入 plan.md
  - 迭代修改：注入 README.md + report.md
  
- **`_get_initial_system_prompt()`**: 首次开发模式的详细提示词
  - 定义开发流程（6个步骤）
  - 代码质量要求
  - 验证要求

- **`_get_iteration_system_prompt()`**: 迭代修改模式的详细提示词
  - 定义修复流程（7个步骤）
  - 问题优先级（致命错误 → 功能缺陷 → 质量问题 → 文档问题）
  - 修改原则（针对性、最小化、立即验证）

- **`_should_break_loop()`**: 循环退出条件
  - 从 report.md 中提取 `accept` 字段
  - 支持正则匹配 `**accept**: true/false` 或 `accept: true/false`

- **`run()`**: 主执行流程
  - 任务执行顺序
  - 循环控制逻辑
  - 自动清理 output 和 memory 目录

### 预期输出

**第一阶段（Researcher + Planner）**:
1. `output/analysis.md`: 分析 Python CLI 开发方案，推荐使用 argparse 和 json 模块
2. `output/plan.md`: 分解为 5 个步骤（数据模型、文件操作、CLI 接口、主程序、文档）

**第二阶段（Loop - 第1轮）**:
3. `output/todo.py`: 主程序代码
4. `output/README.md`: 详细的使用说明
5. `output/report.md`: 首次审查报告
   - 如果 `accept: true` → 完成
   - 如果 `accept: false` → 进入第2轮

**第三阶段（Loop - 第2轮，如果需要）**:
6. 更新 `output/todo.py`: 修复 report.md 中指出的问题
7. 更新 `output/README.md`: 同步更新文档
8. 更新 `output/report.md`: 重新审查
   - 如果 `accept: true` → 完成
   - 如果 `accept: false` → 继续下一轮...

