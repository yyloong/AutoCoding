# MyCursor 架构设计文档

## 概述

MyCursor 是一个基于多 Agent 协作的自动化软件开发系统，通过 4 个专业 Agent 的协同工作，实现从需求分析到代码实现、测试审查的完整开发流程。

## 系统架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户输入需求                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Researcher Agent (调研专家)                                 │
│  - 理解需求                                                   │
│  - 互联网调研                                                 │
│  - 技术选型                                                   │
│  - 项目规模评估                                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ analysis.md
┌─────────────────────────────────────────────────────────────┐
│  Planner Agent (规划师)                                      │
│  - 读取调研报告                                               │
│  - 任务分解                                                   │
│  - 制定开发计划                                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ plan.md
┌─────────────────────────────────────────────────────────────┐
│                    Loop 循环 (最多 10 次)                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Coder Agent (开发工程师)                                │ │
│ │  输入: plan.md + feedback.md (如果有)                    │ │
│ │  - 逐步实现功能                                          │ │
│ │  - 验证代码正确性                                        │ │
│ │  - 生成文档                                              │ │
│ │  输出: 代码文件 + README.md                              │ │
│ └────────────────────┬────────────────────────────────────┘ │
│                      │                                       │
│                      ▼ 代码 + README.md                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Refiner Agent (审查专家)                                │ │
│ │  - 安装依赖                                              │ │
│ │  - 运行代码                                              │ │
│ │  - 测试功能                                              │ │
│ │  - 生成报告                                              │ │
│ │  输出: report.md (含 accept 字段)                        │ │
│ └────────────────────┬────────────────────────────────────┘ │
│                      │                                       │
│                      ▼                                       │
│              accept == true?                                │
│                   /     \                                    │
│                 Yes      No                                  │
│                  │        │                                  │
│                  │        ▼ feedback.md                      │
│                  │        │                                  │
│                  │        └──────┐                           │
│                  │               │                           │
│                  ▼               │                           │
└──────────────── 退出 ◄───────────┘                           │
                                                               │
                  最终交付物                                    │
└─────────────────────────────────────────────────────────────┘
```

## 文件说明

### 配置文件

#### 1. workflow.yaml
主工作流配置文件，定义了整个系统的执行流程。

**关键配置**:
```yaml
type: LoopWorkflow          # 使用循环工作流

researcher:                 # 第一步：调研
  agent_config: researcher.yaml

planner:                    # 第二步：规划
  agent_config: planner.yaml
  
loop:                       # 第三步：循环开发
  coder:                    # 循环内第一个 agent
    agent_config: coder.yaml
    breaker: false          # 不是循环退出点
  
  refiner:                  # 循环内第二个 agent
    agent_config: refiner.yaml
    breaker: true           # 是循环退出点，由它决定是否继续
```

#### 2. researcher.yaml
Researcher Agent 的配置文件。

**关键特性**:
- 模型: `qwen3-coder-flash` (速度优先)
- 温度: 0.3 (较低，保证输出稳定)
- 工具: `web_research`, `file_system`, `exit_task`
- 最大轮数: 20

**提示词设计**:
- 4 步工作流：需求理解 → 互联网调研 → 方案分析 → 生成报告
- 输出格式: 结构化的 `analysis.md`
- 包含：需求描述、技术调研、解决方案、项目规模评估

#### 3. planner.yaml
Planner Agent 的配置文件。

**关键特性**:
- 模型: `qwen3-coder-flash` (速度优先)
- 温度: 0.2 (更低，保证计划稳定)
- 工具: `file_system`, `exit_task`
- 最大轮数: 15

**提示词设计**:
- 5 步工作流：读取报告 → 任务分解 → 制定计划 → 生成大纲 → 退出
- 输出格式: 结构化的 `plan.md`
- 包含：项目概述、技术栈、开发步骤（每步有任务清单、涉及文件、关键技术点、验证标准）

#### 4. coder.yaml
Coder Agent 的配置文件。

**关键特性**:
- 模型: `qwen3-coder-plus` (质量优先)
- 温度: 0.3 (平衡创造性和稳定性)
- 工具: `file_system`, `python_executor`, `code_executor`, `web_search`, `exit_task`
- 最大轮数: 100 (允许多轮迭代)
- 超时: 60 秒

**提示词设计**:
- 6 步工作流：读取计划和反馈 → 分步实现 → 必要时调研 → 生成文档 → 最终验证 → 退出
- 强调：每步都要验证代码能运行
- 代码质量要求：清晰、模块化、有注释、符合规范
- 输出：项目代码 + `README.md`

#### 5. refiner.yaml
Refiner Agent 的配置文件。

**关键特性**:
- 模型: `qwen3-coder-plus` (质量优先)
- 温度: 0.2 (保证审查严谨)
- 工具: `file_system`, `environment_setup`, `code_executor`, `shell_executor`, `exit_task`
- 最大轮数: 50
- 超时: 60 秒

**提示词设计**:
- 8 步工作流：环境准备 → 安装依赖 → 运行项目 → 功能验证 → 问题分析 → 生成报告 → 生成反馈 → 退出
- 问题分类：致命错误、功能缺陷、质量问题、文档问题
- 输出：`report.md` (含 `accept` 字段) + `feedback.md` (如果 accept=false)

### 核心代码文件

#### 6. loop_workflow.py
LoopWorkflow 的实现文件，继承自 `ms_agent.workflow.base.Workflow`。

**关键方法**:

1. **`build_workflow()`**
   - 解析配置文件
   - 构建工作流链

2. **`run(inputs, **kwargs)`**
   - 主执行方法
   - 处理非循环任务（researcher, planner）
   - 处理循环任务（coder, refiner）
   - 控制循环次数

3. **`_inject_plan_and_feedback(inputs)`**
   - 在 coder 运行前注入上下文
   - 读取 `plan.md` 和 `feedback.md`
   - 将内容添加到输入中

4. **`_should_break_loop(outputs)`**
   - 判断是否退出循环
   - 从 refiner 的输出或 `report.md` 中提取 `accept` 字段
   - 返回 True 表示退出循环

**循环逻辑**:
```python
for task_name in itertools.cycle(['coder', 'refiner']):
    if task_name == 'coder':
        # 注入 plan.md 和 feedback.md
        inputs = self._inject_plan_and_feedback(inputs)
        iteration_count += 1
    
    # 运行 agent
    outputs = await agent.run(inputs)
    
    if task_name == 'refiner' and task_info['breaker']:
        if self._should_break_loop(outputs):
            break  # accept=true，退出循环
        else:
            continue  # accept=false，继续下一轮
```

#### 7. simple_agent.py
SimpleAgent 的实现文件，继承自 `ms_agent.LLMAgent`。

**关键方法**:

1. **`step(messages)`**
   - 执行单步交互
   - 生成 LLM 响应
   - 处理工具调用
   - 检测 `exit_task` 工具调用并设置停止标志

**特性**:
- 支持流式输出
- 自动保存历史
- 并行工具调用
- 退出任务检测

### 文档文件

#### 8. README.md
用户使用指南，包含：
- 工作流架构图
- 各 Agent 职责说明
- 循环机制解释
- 配置文件说明
- 使用方法
- 待实现工具列表
- 自定义和扩展指南
- 注意事项
- 示例

#### 9. ARCHITECTURE.md (本文件)
架构设计文档，包含：
- 系统概述
- 整体流程图
- 文件说明
- 数据流分析
- 关键设计决策
- 扩展点

## 数据流分析

### 文件流转

```
用户需求 (字符串)
    ↓
[Researcher]
    ↓
analysis.md
    ├─→ [Planner] 读取
    └─→ [Coder] 可选读取
        ↓
plan.md
    └─→ [Coder] 必读 (由 loop_workflow 注入)
        ↓
代码文件 + README.md
    └─→ [Refiner] 读取并运行
        ↓
report.md (含 accept 字段)
    ├─→ [loop_workflow] 读取 accept 字段
    └─→ 如果 accept=false
            ↓
        feedback.md
            └─→ [Coder] 读取 (由 loop_workflow 注入)
                ↓
            (返回循环开始)
```

### 消息流转

```
用户输入 (Message)
    ↓
[Researcher] 
    ↓ (Message list)
[Planner]
    ↓ (Message list)
┌─→ [Coder] ← (注入 plan.md + feedback.md)
│   ↓ (Message list)
│   [Refiner]
│   ↓ (Message list)
└── 检查 accept 字段
        ├─ true: 退出
        └─ false: 返回 Coder
```

## 关键设计决策

### 1. 为什么使用 LoopWorkflow？

**原因**:
- 开发过程本质上是迭代的
- 一次性生成完美代码不现实
- 需要 coder 和 refiner 的反馈循环

**优势**:
- 自动化迭代过程
- 逐步提升代码质量
- 减少人工干预

### 2. 为什么分离 Researcher 和 Planner？

**原因**:
- 调研和规划是不同的认知任务
- 调研需要发散思维（搜索、探索）
- 规划需要收敛思维（分解、组织）

**优势**:
- 每个 agent 专注于单一职责
- 提示词更简洁、更有针对性
- 便于独立优化和调试

### 3. 为什么 Coder 和 Refiner 使用不同的模型配置？

**Coder**:
- 温度 0.3：需要一定创造性
- 超时 60s：代码生成可能较慢
- 最大轮数 100：允许多轮迭代

**Refiner**:
- 温度 0.2：需要更严谨的判断
- 超时 60s：运行代码可能较慢
- 最大轮数 50：测试流程相对固定

### 4. 为什么使用文件而不是消息传递？

**原因**:
- 文件是持久化的，便于调试
- 文件可以被多个 agent 读取
- 文件格式清晰（Markdown），便于人工检查

**优势**:
- 可追溯性强
- 便于中断和恢复
- 便于人工介入

### 5. 为什么 accept 字段在 report.md 中？

**原因**:
- report.md 是 refiner 的正式输出
- accept 是审查结论的一部分
- 便于 loop_workflow 解析

**设计**:
- 使用简单的 `**accept**: true/false` 格式
- 支持从消息内容或文件中解析
- 默认为 false（保守策略）

## 扩展点

### 1. 添加新的 Agent

**步骤**:
1. 创建新的 yaml 配置文件（参考现有文件）
2. 在 `workflow.yaml` 中添加配置
3. 如果需要在循环中，添加到 `loop` 部分

**示例**:
```yaml
# 添加一个 tester agent
loop:
  coder:
    agent_config: coder.yaml
    breaker: false
  
  tester:  # 新增
    agent_config: tester.yaml
    breaker: false
  
  refiner:
    agent_config: refiner.yaml
    breaker: true
```

### 2. 自定义循环逻辑

**修改点**: `loop_workflow.py` 中的 `run()` 方法

**可定制**:
- 循环顺序（不一定是 coder → refiner）
- 退出条件（不一定是 accept 字段）
- 文件注入逻辑（可以注入更多上下文）
- 迭代计数策略（可以按 agent 分别计数）

### 3. 添加新的工具

**步骤**:
1. 实现工具类（继承自 `ms_agent.tool.Tool`）
2. 在 agent 的 yaml 中添加工具配置
3. 在提示词中说明工具的使用方法

**示例**:
```yaml
tools:
  my_custom_tool:
    mcp: false
    # 自定义配置
```

### 4. 修改输出格式

**修改点**: 各 agent 的 `prompt.system` 中的输出格式说明

**建议**:
- 保持结构化（Markdown、JSON 等）
- 包含必要的元数据（时间、版本等）
- 便于机器解析

### 5. 集成外部系统

**可能的集成点**:
- Git 版本控制（在 coder 中提交代码）
- CI/CD 系统（在 refiner 中触发测试）
- 问题跟踪系统（记录 feedback）
- 代码审查平台（发布 report）

## 待办事项

### 高优先级
- [ ] 实现 `python_executor` 工具
- [ ] 实现 `code_executor` 工具
- [ ] 实现 `environment_setup` 工具
- [ ] 实现 `shell_executor` 工具

### 中优先级
- [ ] 添加单元测试
- [ ] 添加集成测试
- [ ] 优化提示词（基于实际使用反馈）
- [ ] 添加日志和监控

### 低优先级
- [ ] 支持更多编程语言
- [ ] 添加 Web UI
- [ ] 支持并行开发多个模块
- [ ] 添加成本估算功能

## 性能考虑

### Token 消耗
- Researcher: ~5K tokens (调研 + 报告)
- Planner: ~3K tokens (规划)
- Coder (每轮): ~20K tokens (代码生成)
- Refiner (每轮): ~10K tokens (测试 + 报告)

**估算**: 一个中型项目，3 轮循环，总计约 100K tokens

### 时间消耗
- Researcher: ~2-3 分钟（取决于搜索）
- Planner: ~1-2 分钟
- Coder (每轮): ~5-10 分钟（取决于代码量）
- Refiner (每轮): ~3-5 分钟（取决于测试复杂度）

**估算**: 一个中型项目，3 轮循环，总计约 30-45 分钟

### 优化建议
1. 使用缓存避免重复调用
2. 并行执行独立任务（如果有）
3. 使用更快的模型（如 flash）处理简单任务
4. 限制最大循环次数避免无限循环

## 故障处理

### 常见问题

1. **循环不退出**
   - 检查 refiner 的 accept 判断标准
   - 检查 report.md 的格式是否正确
   - 增加日志查看 `_should_break_loop` 的返回值

2. **代码质量不佳**
   - 调整 coder 的温度参数
   - 优化 coder 的提示词
   - 增加代码验证步骤

3. **工具调用失败**
   - 检查工具是否正确实现
   - 检查超时设置是否合理
   - 查看工具调用日志

4. **文件读写错误**
   - 检查 output_dir 权限
   - 检查文件路径是否正确
   - 确保文件编码为 UTF-8

## 总结

MyCursor 是一个设计良好、可扩展的多 Agent 协作系统。通过清晰的职责分离、结构化的数据流、和灵活的循环机制，它能够自动化完成从需求到代码的完整开发流程。

**核心优势**:
- 🎯 职责明确：每个 agent 专注于单一任务
- 🔄 自动迭代：通过循环机制持续改进
- 📝 可追溯：所有中间产物都以文件形式保存
- 🔧 可扩展：易于添加新 agent 和工具
- 🚀 高效：合理的模型选择和并行策略

**适用场景**:
- 小型到中型项目的快速原型开发
- 学习项目和示例代码生成
- 代码重构和优化
- 技术方案验证

**不适用场景**:
- 超大型企业级系统
- 需要复杂架构设计的项目
- 对性能和安全有极高要求的项目
- 需要大量人工决策的项目

