# Graph 模块详解 - DeerFlow 工作流引擎核心

## 一、模块概述

### 1.1 Graph 模块的定位

Graph 模块是 DeerFlow 项目的**核心工作流引擎**，基于 LangGraph 框架构建，负责：
- 定义和管理整个研究流程的状态机
- 协调各个智能体节点的执行
- 管理状态的流转和持久化
- 提供工作流的构建和编译接口

### 1.2 目录结构

```
src/graph/
├── __init__.py          # 模块导出接口
├── builder.py           # 工作流图构建器
├── types.py             # 状态类型定义
├── nodes.py             # 工作流节点实现
├── utils.py             # 工具函数
└── checkpoint.py        # 检查点和流式消息管理
```

### 1.3 核心职责

| 文件 | 核心职责 | 关键类/函数 |
|------|---------|------------|
| `builder.py` | 构建 LangGraph 工作流图 | `build_graph()`, `build_graph_with_memory()` |
| `types.py` | 定义工作流状态结构 | `State` 类 |
| `nodes.py` | 实现各个工作流节点 | 各种 `*_node()` 函数 |
| `utils.py` | 提供辅助工具函数 | 消息处理、澄清历史构建 |
| `checkpoint.py` | 管理检查点和流式输出 | `ChatStreamManager` 类 |

---

## 二、文件详解

### 2.1 types.py - 状态类型定义

**文件路径：** `src/graph/types.py`

#### 2.1.1 核心类：State

```python
class State(MessagesState):
    """工作流状态类，继承自 LangGraph 的 MessagesState"""
```

**继承关系：**
- 继承自 `MessagesState`（LangGraph 提供）
- 自动包含 `messages` 字段用于存储对话历史

#### 2.1.2 状态字段分类

**运行时变量（Runtime Variables）：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `locale` | str | "en-US" | 用户语言区域设置 |
| `research_topic` | str | "" | 原始研究主题 |
| `clarified_research_topic` | str | "" | 澄清后的完整主题 |
| `observations` | list[str] | [] | 观察结果列表 |
| `resources` | list[Resource] | [] | 资源列表 |
| `plan_iterations` | int | 0 | 计划迭代次数 |
| `current_plan` | Plan \| str | None | 当前研究计划 |
| `final_report` | str | "" | 最终报告 |
| `auto_accepted_plan` | bool | False | 计划是否自动接受 |
| `enable_background_investigation` | bool | True | 是否启用背景调查 |
| `background_investigation_results` | str | None | 背景调查结果 |

**引用元数据（Citation Metadata）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `citations` | list[dict[str, Any]] | 研究过程中收集的引用信息 |

**澄清状态追踪（Clarification State）：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_clarification` | bool | False | 是否启用澄清功能 |
| `clarification_rounds` | int | 0 | 当前澄清轮次 |
| `clarification_history` | list[str] | [] | 澄清历史记录 |
| `is_clarification_complete` | bool | False | 澄清是否完成 |
| `max_clarification_rounds` | int | 3 | 最大澄清轮次 |

**工作流控制（Workflow Control）：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `goto` | str | "planner" | 下一个要执行的节点 |

#### 2.1.3 状态设计特点

1. **类型安全**：使用 Python 类型注解，确保字段类型正确
2. **可扩展**：支持添加新字段而不影响现有功能
3. **可序列化**：所有字段都可以序列化，支持检查点持久化
4. **分类清晰**：字段按功能分组，便于理解和维护

---

### 2.2 builder.py - 工作流图构建器

**文件路径：** `src/graph/builder.py`

#### 2.2.1 核心功能

Builder 模块负责构建 LangGraph 工作流图，定义节点和边的连接关系。

**主要函数：**

| 函数 | 功能 | 返回值 |
|------|------|--------|
| `_build_base_graph()` | 构建基础工作流图 | StateGraph |
| `build_graph()` | 构建无内存的工作流图 | CompiledGraph |
| `build_graph_with_memory()` | 构建带内存的工作流图 | CompiledGraph |
| `continue_to_running_research_team()` | 条件路由函数 | str |

#### 2.2.2 工作流图结构

```python
def _build_base_graph():
    """构建基础状态图，包含所有节点和边"""
    builder = StateGraph(State)

    # 添加节点
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)

    # 添加边
    builder.add_edge(START, "coordinator")
    builder.add_edge("background_investigator", "planner")
    builder.add_edge("reporter", END)

    # 添加条件边
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "analyst", "coder"]
    )

    return builder
```

#### 2.2.3 节点说明

**核心节点：**

| 节点名称 | 对应函数 | 功能 |
|---------|---------|------|
| `coordinator` | `coordinator_node` | 协调器，处理用户输入和澄清 |
| `background_investigator` | `background_investigation_node` | 背景调查，初步搜索 |
| `planner` | `planner_node` | 规划器，生成研究计划 |
| `research_team` | `research_team_node` | 研究团队协调器 |
| `researcher` | `researcher_node` | 研究员，执行搜索和爬虫 |
| `analyst` | `analyst_node` | 分析员，综合信息 |
| `coder` | `coder_node` | 编码员，执行代码 |
| `human_feedback` | `human_feedback_node` | 人工反馈节点 |
| `reporter` | `reporter_node` | 报告生成器 |

#### 2.2.4 条件路由逻辑

```python
def continue_to_running_research_team(state: State):
    """
    根据当前计划状态决定下一个执行的节点

    路由逻辑：
    1. 如果没有计划或计划为空 → 返回 "planner"
    2. 如果所有步骤都已完成 → 返回 "planner"
    3. 找到第一个未完成的步骤：
       - RESEARCH 类型 → 返回 "researcher"
       - ANALYSIS 类型 → 返回 "analyst"
       - PROCESSING 类型 → 返回 "coder"
    4. 其他情况 → 返回 "planner"
    """
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return "planner"

    if all(step.execution_res for step in current_plan.steps):
        return "planner"

    # 找到第一个未完成的步骤
    incomplete_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            incomplete_step = step
            break

    if not incomplete_step:
        return "planner"

    # 根据步骤类型路由
    if incomplete_step.step_type == StepType.RESEARCH:
        return "researcher"
    if incomplete_step.step_type == StepType.ANALYSIS:
        return "analyst"
    if incomplete_step.step_type == StepType.PROCESSING:
        return "coder"
    return "planner"
```

#### 2.2.5 内存管理

**无内存模式：**
```python
def build_graph():
    """构建无内存的工作流图"""
    builder = _build_base_graph()
    return builder.compile()
```

**带内存模式：**
```python
def build_graph_with_memory():
    """构建带内存的工作流图"""
    memory = MemorySaver()  # 使用内存保存器
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)
```

**区别：**
- 无内存模式：每次执行都是独立的，不保存状态
- 带内存模式：支持检查点，可以中断和恢复

---

### 2.3 utils.py - 工具函数

**文件路径：** `src/graph/utils.py`

#### 2.3.1 核心功能

Utils 模块提供消息处理和澄清历史管理的辅助函数。

#### 2.3.2 消息处理函数

**1. get_message_content()**

```python
def get_message_content(message: Any) -> str:
    """从字典或 LangChain 消息对象中提取内容"""
```

**功能：**
- 支持字典格式：`message.get("content", "")`
- 支持对象格式：`getattr(message, "content", "")`
- 统一的消息内容提取接口

**2. is_user_message()**

```python
def is_user_message(message: Any) -> bool:
    """判断消息是否来自用户"""
```

**判断逻辑：**
1. 检查 `role` 字段是否为 "user" 或 "human"
2. 检查 `type` 字段是否为 "human"
3. 检查 `name` 字段是否不在助手名称列表中
4. 支持字典和对象两种格式

**助手名称列表：**
```python
ASSISTANT_SPEAKER_NAMES = {
    "coordinator",
    "planner",
    "researcher",
    "coder",
    "reporter",
    "background_investigator",
}
```

**3. get_latest_user_message()**

```python
def get_latest_user_message(messages: list[Any]) -> tuple[Any, str]:
    """获取最新的用户消息及其内容"""
```

**功能：**
- 从消息列表末尾开始反向遍历
- 找到第一个用户消息
- 返回消息对象和内容的元组

#### 2.3.3 澄清历史管理函数

**1. build_clarified_topic_from_history()**

```python
def build_clarified_topic_from_history(
    clarification_history: list[str]
) -> tuple[str, list[str]]:
    """从澄清历史构建完整的澄清主题字符串"""
```

**构建逻辑：**
- 单轮澄清：直接返回该内容
- 多轮澄清：格式化为 "初始主题 - 澄清1, 澄清2, ..."

**示例：**
```python
history = ["量子计算", "应用场景", "医疗领域"]
result = "量子计算 - 应用场景, 医疗领域"
```

**2. reconstruct_clarification_history()**

```python
def reconstruct_clarification_history(
    messages: list[Any],
    fallback_history: list[str] | None = None,
    base_topic: str = ""
) -> list[str]:
    """从消息列表重建澄清历史"""
```

**重建逻辑：**
1. 遍历所有消息，提取用户消息内容
2. 去除连续重复的内容
3. 如果没有找到用户消息，使用 fallback_history
4. 如果 fallback 也为空，使用 base_topic

**用途：**
- 从对话历史恢复澄清状态
- 支持会话恢复和检查点加载

---

### 2.4 checkpoint.py - 检查点和流式消息管理

**文件路径：** `src/graph/checkpoint.py`

#### 2.4.1 核心类：ChatStreamManager

```python
class ChatStreamManager:
    """管理聊天流式消息的持久化存储和内存缓存"""
```

**职责：**
- 处理流式消息的分块存储
- 支持 MongoDB 和 PostgreSQL 持久化
- 提供内存缓存机制
- 管理会话的完整生命周期

#### 2.4.2 初始化配置

```python
def __init__(
    self,
    checkpoint_saver: bool = False,
    db_uri: Optional[str] = None
) -> None:
```

**参数：**
- `checkpoint_saver`: 是否启用检查点保存
- `db_uri`: 数据库连接 URI

**支持的数据库：**
- MongoDB: `mongodb://localhost:27017`
- PostgreSQL: `postgresql://user:pass@host/db`

#### 2.4.3 核心方法

**1. process_stream_message()**

```python
def process_stream_message(
    self,
    thread_id: str,
    message: str,
    finish_reason: str
) -> bool:
```

**功能：**
- 处理单个消息块
- 存储到内存缓存
- 当流结束时触发持久化

**finish_reason 类型：**
- `"stop"`: 正常结束，触发持久化
- `"interrupt"`: 中断结束，触发持久化
- 其他: 继续缓存，不持久化

**处理流程：**
```
接收消息块
    ↓
获取/初始化游标
    ↓
存储到内存 (InMemoryStore)
    ↓
检查 finish_reason
    ↓
如果是 stop/interrupt
    ↓
持久化到数据库
    ↓
清理内存缓存
```

**2. _persist_complete_conversation()**

```python
def _persist_complete_conversation(
    self,
    thread_id: str,
    store_namespace: Tuple[str, str],
    final_index: int
) -> bool:
```

**功能：**
- 从内存中检索所有消息块
- 合并成完整对话
- 持久化到数据库

**3. _persist_to_mongodb()**

```python
def _persist_to_mongodb(
    self,
    thread_id: str,
    messages: List[str]
) -> bool:
```

**MongoDB 存储结构：**
```json
{
    "thread_id": "unique_thread_id",
    "messages": ["msg1", "msg2", "..."],
    "ts": "2026-02-11T10:00:00",
    "id": "uuid"
}
```

**4. _persist_to_postgresql()**

```python
def _persist_to_postgresql(
    self,
    thread_id: str,
    messages: List[str]
) -> bool:
```

**PostgreSQL 表结构：**
```sql
CREATE TABLE chat_streams (
    id UUID PRIMARY KEY,
    thread_id VARCHAR(255) UNIQUE,
    messages JSONB,
    ts TIMESTAMP WITH TIME ZONE
);
```

#### 2.4.4 使用示例

```python
# 创建管理器
manager = ChatStreamManager(
    checkpoint_saver=True,
    db_uri="mongodb://localhost:27017"
)

# 处理流式消息
manager.process_stream_message(
    thread_id="thread_123",
    message="消息块1",
    finish_reason="partial"
)

manager.process_stream_message(
    thread_id="thread_123",
    message="消息块2",
    finish_reason="stop"  # 触发持久化
)

# 关闭连接
manager.close()
```

---

### 2.5 nodes.py - 工作流节点实现

**文件路径：** `src/graph/nodes.py`

#### 2.5.1 节点概述

Nodes 模块实现了工作流中的所有节点函数，每个节点负责特定的任务。

**节点分类：**

| 类别 | 节点 | 功能 |
|------|------|------|
| **协调节点** | `coordinator_node` | 处理用户输入、澄清、语言检测 |
| **调查节点** | `background_investigation_node` | 执行背景调查和初步搜索 |
| **规划节点** | `planner_node` | 生成和更新研究计划 |
| **执行节点** | `research_team_node` | 协调研究团队执行 |
| | `researcher_node` | 执行搜索和爬虫任务 |
| | `analyst_node` | 执行分析和综合任务 |
| | `coder_node` | 执行代码和数据处理任务 |
| **反馈节点** | `human_feedback_node` | 处理人工反馈 |
| **报告节点** | `reporter_node` | 生成最终报告 |

#### 2.5.2 工具函数

**Handoff 工具：**

```python
@tool
def handoff_to_planner(research_topic: str, locale: str):
    """移交给规划器进行计划"""
    pass

@tool
def handoff_after_clarification(locale: str, research_topic: str):
    """澄清完成后移交"""
    pass

@tool
def direct_response(message: str, locale: str):
    """直接响应用户（用于问候、闲聊）"""
    pass
```

**辅助函数：**

```python
def needs_clarification(state: dict) -> bool:
    """检查是否需要澄清"""
    # 判断逻辑：
    # - 启用澄清功能
    # - 有澄清轮次
    # - 未完成
    # - 未超过最大轮次
    pass

def preserve_state_meta_fields(state: State) -> dict:
    """保留状态元字段"""
    # 提取需要在状态转换中保留的配置字段
    pass
```

#### 2.5.3 节点实现模式

**典型节点结构：**

```python
def example_node(state: State) -> Command[Literal["next_node"]]:
    """节点函数示例"""

    # 1. 提取状态信息
    current_data = state.get("field_name")

    # 2. 执行节点逻辑
    # - 调用 LLM
    # - 使用工具
    # - 处理数据
    result = process_logic(current_data)

    # 3. 更新状态
    updates = {
        "field_name": result,
        "goto": "next_node"
    }

    # 4. 返回命令
    return Command(
        update=updates,
        goto="next_node"
    )
```

**关键特点：**
- 接收 `State` 对象作为输入
- 返回 `Command` 对象指定下一步
- 通过 `update` 字段更新状态
- 通过 `goto` 字段指定路由

---

## 三、模块间关系与交互

### 3.1 依赖关系图

```
builder.py (工作流构建)
    ↓
    ├─→ types.py (状态定义)
    ├─→ nodes.py (节点实现)
    └─→ checkpoint.py (检查点管理)

nodes.py (节点实现)
    ↓
    ├─→ types.py (状态类型)
    ├─→ utils.py (工具函数)
    └─→ 外部模块 (agents, tools, llms)

checkpoint.py (检查点管理)
    ↓
    └─→ 数据库 (MongoDB/PostgreSQL)
```

### 3.2 数据流转

```
用户输入
    ↓
State 初始化 (types.py)
    ↓
Graph 构建 (builder.py)
    ↓
节点执行 (nodes.py)
    ├─→ 使用工具函数 (utils.py)
    └─→ 保存检查点 (checkpoint.py)
    ↓
State 更新
    ↓
下一个节点
```

### 3.3 关键交互点

**1. Builder → Nodes**
- Builder 注册节点函数
- 定义节点间的连接关系
- 配置条件路由

**2. Nodes → State**
- 节点读取状态信息
- 节点更新状态字段
- 通过 Command 返回更新

**3. Nodes → Utils**
- 使用消息处理函数
- 构建澄清历史
- 提取用户消息

**4. Checkpoint → Database**
- 流式消息缓存
- 完整对话持久化
- 会话恢复支持

---

## 四、工作流执行流程

### 4.1 完整执行流程

```
1. 初始化
   ↓
   build_graph() 或 build_graph_with_memory()
   ↓
   创建 State 对象

2. 启动工作流
   ↓
   START → coordinator_node
   ↓
   处理用户输入、语言检测、澄清

3. 背景调查（可选）
   ↓
   background_investigation_node
   ↓
   初步搜索和信息收集

4. 计划生成
   ↓
   planner_node
   ↓
   生成研究计划

5. 研究执行
   ↓
   research_team_node
   ↓
   条件路由到：
   - researcher_node (搜索任务)
   - analyst_node (分析任务)
   - coder_node (代码任务)
   ↓
   循环执行直到所有步骤完成

6. 报告生成
   ↓
   reporter_node
   ↓
   生成最终报告

7. 结束
   ↓
   END
```

### 4.2 状态流转示例

```python
# 初始状态
{
    "research_topic": "量子计算",
    "locale": "zh-CN",
    "messages": [],
    "current_plan": None,
    ...
}

# 经过 coordinator_node
{
    "research_topic": "量子计算",
    "clarified_research_topic": "量子计算的基本原理和应用",
    "goto": "background_investigator",
    ...
}

# 经过 planner_node
{
    "current_plan": {
        "steps": [
            {"step_type": "RESEARCH", "description": "..."},
            {"step_type": "ANALYSIS", "description": "..."}
        ]
    },
    "goto": "research_team",
    ...
}

# 经过 reporter_node
{
    "final_report": "# 量子计算研究报告\n\n...",
    "goto": "END",
    ...
}
```

---

## 五、架构设计总结

### 5.1 设计优势

**1. 清晰的职责分离**
- `types.py` - 纯数据定义
- `builder.py` - 工作流构建
- `nodes.py` - 业务逻辑实现
- `utils.py` - 通用工具
- `checkpoint.py` - 持久化管理

**2. 灵活的状态管理**
- 基于 LangGraph 的状态机
- 类型安全的状态定义
- 支持检查点和恢复
- 可扩展的字段设计

**3. 强大的路由机制**
- 条件路由支持
- 动态节点选择
- 基于计划类型的智能分发

**4. 完善的持久化**
- 支持多种数据库
- 流式消息处理
- 自动清理机制

### 5.2 关键技术点

| 技术 | 应用 | 优势 |
|------|------|------|
| LangGraph | 工作流引擎 | 状态管理、可视化、检查点 |
| TypedDict | 状态定义 | 类型安全、IDE 支持 |
| Command 模式 | 节点返回 | 明确的控制流 |
| InMemoryStore | 消息缓存 | 高性能临时存储 |
| MongoDB/PostgreSQL | 持久化 | 灵活的数据库选择 |

### 5.3 扩展建议

**添加新节点：**
1. 在 `nodes.py` 中实现节点函数
2. 在 `builder.py` 中注册节点
3. 配置节点间的连接关系
4. 更新条件路由逻辑（如需要）

**扩展状态字段：**
1. 在 `types.py` 的 `State` 类中添加字段
2. 提供默认值和类型注解
3. 在相关节点中使用新字段

**自定义持久化：**
1. 继承 `ChatStreamManager`
2. 实现自定义的 `_persist_to_*` 方法
3. 配置数据库连接

---

## 六、参考资料

### 6.1 相关文档

**LangGraph 官方文档：**
- [LangGraph 概述](https://langchain-ai.github.io/langgraph/)
- [StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [Checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/)

**项目内部文档：**
- [01_项目整体脉络和快速上手.md](01_项目整体脉络和快速上手.md)
- [03_整体架构设计和模块关系.md](03_整体架构设计和模块关系.md)

### 6.2 关键文件路径

```
src/graph/
├── __init__.py          # 导出 build_graph 函数
├── builder.py           # 工作流图构建（92 行）
├── types.py             # 状态定义（49 行）
├── nodes.py             # 节点实现（大型文件）
├── utils.py             # 工具函数（114 行）
└── checkpoint.py        # 检查点管理（394 行）
```

### 6.3 使用示例

**基本使用：**
```python
from src.graph import build_graph

# 构建工作流图
graph = build_graph()

# 执行工作流
result = graph.invoke({
    "research_topic": "量子计算",
    "locale": "zh-CN"
})

# 获取最终报告
final_report = result["final_report"]
```

**带内存使用：**
```python
from src.graph import build_graph_with_memory

# 构建带内存的工作流图
graph = build_graph_with_memory()

# 执行工作流（支持检查点）
config = {"configurable": {"thread_id": "thread_123"}}
result = graph.invoke(
    {"research_topic": "量子计算"},
    config=config
)
```

---

## 七、总结

Graph 模块是 DeerFlow 的核心工作流引擎，通过清晰的模块划分和强大的状态管理，实现了灵活、可扩展的多智能体研究流程。

**核心价值：**
- 基于 LangGraph 的成熟工作流框架
- 类型安全的状态管理
- 灵活的节点路由机制
- 完善的持久化支持
- 易于扩展和定制

**适用场景：**
- 多步骤的研究任务
- 需要人工反馈的工作流
- 长时间运行的任务（支持检查点）
- 需要状态追踪的复杂流程

---

**文档版本：** v1.0
**最后更新：** 2026-02-11
**适用项目版本：** DeerFlow latest
