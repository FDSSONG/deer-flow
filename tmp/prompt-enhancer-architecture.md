# Prompt Enhancer 模块架构文档

## 概述

`src/prompt_enhancer` 模块是 Deer-Flow 项目中负责提示词（Prompt）增强的核心模块。该模块使用 LangGraph 框架构建工作流，通过 AI 模型自动优化和增强用户输入的提示词，使其更加清晰、具体和有效。

## 目录结构

```
src/prompt_enhancer/
├── __init__.py                    # 模块初始化文件
└── graph/
    ├── state.py                   # 状态定义
    ├── enhancer_node.py           # 核心增强节点
    └── builder.py                 # 图构建器

tests/unit/prompt_enhancer/
└── graph/
    ├── test_state.py              # 状态测试
    ├── test_enhancer_node.py      # 节点测试
    └── test_builder.py            # 构建器测试
```

## 核心模块详解

### 1. state.py - 状态定义

定义了 Prompt Enhancer 工作流的状态结构。

#### PromptEnhancerState

```python
class PromptEnhancerState(TypedDict):
    """State for the prompt enhancer workflow."""

    prompt: str                      # 原始提示词（必需）
    context: Optional[str]           # 额外上下文信息（可选）
    report_style: Optional[ReportStyle]  # 报告风格偏好（可选）
    output: Optional[str]            # 增强后的提示词结果（可选）
```

**字段说明：**

- **prompt**: 用户输入的原始提示词，需要被增强的内容
- **context**: 额外的上下文信息，帮助 AI 更好地理解用户意图
- **report_style**: 报告风格配置，影响增强后的提示词风格
- **output**: 增强后的提示词结果，由 enhancer_node 生成

### 2. enhancer_node.py - 核心增强节点

这是 Prompt Enhancer 模块的核心处理节点，负责调用 LLM 增强用户提示词。

#### prompt_enhancer_node(state: PromptEnhancerState)

核心节点函数，执行提示词增强逻辑。

**函数签名：**
```python
def prompt_enhancer_node(state: PromptEnhancerState) -> dict
```

**处理流程：**

1. **获取 LLM 模型**
   ```python
   model = get_llm_by_type(AGENT_LLM_MAP["prompt_enhancer"])
   ```
   从配置中获取专门用于提示词增强的 LLM 模型。

2. **构建上下文信息**
   ```python
   context_info = ""
   if state.get("context"):
       context_info = f"\n\nAdditional context: {state['context']}"
   ```
   如果提供了额外上下文，将其添加到请求中。

3. **创建消息**
   ```python
   original_prompt_message = HumanMessage(
       content=f"Please enhance this prompt:{context_info}\n\nOriginal prompt: {state['prompt']}"
   )
   ```
   构建包含原始提示词和上下文的消息。

4. **应用提示词模板**
   ```python
   messages = apply_prompt_template(
       "prompt_enhancer/prompt_enhancer",
       {
           "messages": [original_prompt_message],
           "report_style": state.get("report_style"),
       },
       locale=state.get("locale", "en-US"),
   )
   ```
   使用预定义的模板格式化消息，支持国际化和报告风格。

5. **调用 LLM**
   ```python
   response = model.invoke(messages)
   response_content = response.content.strip()
   ```
   发送请求到 LLM 并获取响应。

6. **解析响应（双重策略）**

   **策略 1: XML 标签提取（优先）**
   ```python
   xml_match = re.search(
       r"<enhanced_prompt>(.*?)</enhanced_prompt>",
       response_content,
       re.DOTALL
   )
   if xml_match:
       enhanced_prompt = xml_match.group(1).strip()
   ```
   优先尝试从 XML 标签中提取增强后的提示词。

   **策略 2: 回退解析**
   ```python
   else:
       enhanced_prompt = response_content
       # 移除常见前缀
       prefixes_to_remove = [
           "Enhanced Prompt:",
           "Enhanced prompt:",
           "Here's the enhanced prompt:",
           ...
       ]
   ```
   如果没有 XML 标签，使用回退逻辑移除常见前缀。

7. **返回结果**
   ```python
   return {"output": enhanced_prompt}
   ```
   返回增强后的提示词。

**错误处理：**
```python
except Exception as e:
    logger.error(f"Error in prompt enhancement: {str(e)}")
    return {"output": state["prompt"]}
```
如果增强失败，返回原始提示词，确保系统稳定性。

### 3. builder.py - 图构建器

使用 LangGraph 框架构建提示词增强工作流。

#### build_graph()

构建并返回提示词增强工作流图。

**函数签名：**
```python
def build_graph() -> CompiledGraph
```

**构建步骤：**

1. **创建状态图**
   ```python
   builder = StateGraph(PromptEnhancerState)
   ```
   使用 `PromptEnhancerState` 作为状态类型创建图构建器。

2. **添加节点**
   ```python
   builder.add_node("enhancer", prompt_enhancer_node)
   ```
   添加名为 "enhancer" 的节点，绑定到 `prompt_enhancer_node` 函数。

3. **设置入口点**
   ```python
   builder.set_entry_point("enhancer")
   ```
   将 "enhancer" 节点设置为工作流的入口点。

4. **设置结束点**
   ```python
   builder.set_finish_point("enhancer")
   ```
   将 "enhancer" 节点设置为工作流的结束点。

5. **编译图**
   ```python
   return builder.compile()
   ```
   编译并返回可执行的工作流图。

**工作流结构：**
```
START → enhancer → END
```

这是一个简单的单节点工作流，专注于提示词增强这一核心功能。

## 架构设计

### 设计模式

#### 1. 状态机模式

使用 LangGraph 的 StateGraph 实现状态机模式：

```
输入状态 (PromptEnhancerState)
    ↓
处理节点 (prompt_enhancer_node)
    ↓
输出状态 (PromptEnhancerState with output)
```

**优势：**
- 状态流转清晰可追踪
- 易于扩展和维护
- 支持复杂的工作流编排

#### 2. 模板方法模式

通过 `apply_prompt_template` 使用预定义的模板：

```python
messages = apply_prompt_template(
    "prompt_enhancer/prompt_enhancer",
    {...},
    locale=state.get("locale", "en-US")
)
```

**优势：**
- 统一的提示词格式
- 支持国际化
- 易于调整和优化

#### 3. 策略模式

响应解析使用双重策略：

```
策略 1: XML 标签提取 (优先)
    ↓ (失败)
策略 2: 前缀移除回退
```

**优势：**
- 提高解析成功率
- 兼容不同的 LLM 响应格式
- 降低解析失败风险

### 数据流

#### 完整数据流程

```
用户输入
  ↓
{
  prompt: "原始提示词",
  context: "上下文信息",
  report_style: ReportStyle,
  locale: "zh-CN"
}
  ↓
build_graph() → 创建工作流
  ↓
graph.invoke(state) → 执行工作流
  ↓
prompt_enhancer_node(state)
  ↓
获取 LLM 模型 (AGENT_LLM_MAP["prompt_enhancer"])
  ↓
构建消息 (HumanMessage + context)
  ↓
应用模板 (apply_prompt_template)
  ↓
调用 LLM (model.invoke)
  ↓
解析响应
  ├─ XML 标签提取 (优先)
  └─ 前缀移除回退
  ↓
返回结果
  ↓
{
  prompt: "原始提示词",
  context: "上下文信息",
  report_style: ReportStyle,
  output: "增强后的提示词"
}
```

## 使用示例

### 1. 基础使用

```python
from src.prompt_enhancer.graph.builder import build_graph

# 构建工作流图
graph = build_graph()

# 准备输入状态
input_state = {
    "prompt": "写一篇关于 AI 的文章"
}

# 执行工作流
result = graph.invoke(input_state)

# 获取增强后的提示词
enhanced_prompt = result["output"]
print(enhanced_prompt)
```

### 2. 带上下文的增强

```python
from src.prompt_enhancer.graph.builder import build_graph

graph = build_graph()

input_state = {
    "prompt": "分析这个问题",
    "context": "这是一个关于机器学习模型性能优化的技术问题，需要考虑训练速度和准确率的平衡"
}

result = graph.invoke(input_state)
enhanced_prompt = result["output"]
```

### 3. 指定报告风格

```python
from src.prompt_enhancer.graph.builder import build_graph
from src.config.report_style import ReportStyle

graph = build_graph()

input_state = {
    "prompt": "研究量子计算",
    "report_style": ReportStyle.ACADEMIC,  # 学术风格
    "locale": "zh-CN"  # 中文
}

result = graph.invoke(input_state)
enhanced_prompt = result["output"]
```

### 4. 异步执行

```python
import asyncio
from src.prompt_enhancer.graph.builder import build_graph

async def enhance_prompt_async(prompt: str):
    graph = build_graph()
    result = await graph.ainvoke({"prompt": prompt})
    return result["output"]

# 并发增强多个提示词
prompts = ["提示词1", "提示词2", "提示词3"]
results = await asyncio.gather(*[enhance_prompt_async(p) for p in prompts])
```

## 配置说明

### LLM 模型配置

Prompt Enhancer 使用的 LLM 模型通过 `AGENT_LLM_MAP` 配置：

```python
# src/config/agents.py
AGENT_LLM_MAP = {
    "prompt_enhancer": "basic",  # 使用 basic 类型的 LLM
    ...
}
```

可以根据需要修改为其他 LLM 类型：
- `"basic"`: 基础模型（默认）
- `"reasoning"`: 推理模型（更强的逻辑能力）
- `"code"`: 代码模型
- `"vision"`: 视觉模型

### 提示词模板

模板位置：`src/prompts/prompt_enhancer/prompt_enhancer.{locale}.md`

支持的语言：
- `en-US`: 英文
- `zh-CN`: 中文
- 其他自定义语言

## 最佳实践

### 1. 提供清晰的上下文

```python
# 推荐：提供详细上下文
input_state = {
    "prompt": "优化代码",
    "context": "这是一个 Python 数据处理脚本，主要问题是处理大文件时内存占用过高"
}

# 不推荐：缺少上下文
input_state = {
    "prompt": "优化代码"
}
```

<thinking>
用户再次强调了要分段输出内容。我已经添加了一部分内容，现在继续添加更多的最佳实践和注意事项。我会继续使用 Edit 工具分段添加。
</thinking>

### 2. 复用工作流图

```python
# 推荐：复用图实例
graph = build_graph()
for prompt in prompts:
    result = graph.invoke({"prompt": prompt})

# 不推荐：每次都重新构建
for prompt in prompts:
    graph = build_graph()  # 不必要的开销
    result = graph.invoke({"prompt": prompt})
```

### 3. 错误处理

```python
from src.prompt_enhancer.graph.builder import build_graph

graph = build_graph()

try:
    result = graph.invoke({"prompt": user_input})
    enhanced = result["output"]
except Exception as e:
    logger.error(f"Prompt enhancement failed: {e}")
    enhanced = user_input  # 回退到原始输入
```

### 4. 日志记录

模块内置了详细的日志记录：

```python
import logging

# 启用调试日志查看详细信息
logging.getLogger("src.prompt_enhancer").setLevel(logging.DEBUG)
```

日志级别：
- `INFO`: 增强开始和完成
- `DEBUG`: 响应内容和解析过程
- `WARNING`: 回退解析警告
- `ERROR`: 增强失败错误

## 注意事项

### 1. 响应格式

模块优先解析 XML 标签格式的响应：

```xml
<enhanced_prompt>
增强后的提示词内容
</enhanced_prompt>
```

如果 LLM 没有返回 XML 标签，会使用回退解析逻辑。

### 2. 回退机制

当增强失败时，系统会返回原始提示词，确保不会中断用户流程：

```python
except Exception as e:
    logger.error(f"Error in prompt enhancement: {str(e)}")
    return {"output": state["prompt"]}  # 返回原始提示词
```

### 3. 性能考虑

- 每次增强都会调用 LLM，有一定延迟
- 建议对频繁使用的提示词进行缓存
- 异步调用可以提高并发性能

### 4. Token 消耗

提示词增强会消耗额外的 token：
- 输入：原始提示词 + 上下文 + 系统提示
- 输出：增强后的提示词

建议监控 token 使用情况。

## 技术亮点

### 1. 简洁的架构设计

单节点工作流设计，专注于核心功能：

```
START → enhancer → END
```

**优势：**
- 代码简洁易懂
- 维护成本低
- 执行效率高

### 2. 双重解析策略

结合 XML 标签提取和前缀移除回退：

```python
# 策略 1: XML 标签（结构化）
<enhanced_prompt>内容</enhanced_prompt>

# 策略 2: 前缀移除（兼容性）
Enhanced Prompt: 内容
```

**优势：**
- 提高解析成功率
- 兼容不同 LLM 响应格式
- 降低失败风险

### 3. 国际化支持

通过 `locale` 参数支持多语言：

```python
messages = apply_prompt_template(
    "prompt_enhancer/prompt_enhancer",
    {...},
    locale=state.get("locale", "en-US")
)
```

### 4. 灵活的配置

- 支持自定义 LLM 模型（通过 AGENT_LLM_MAP）
- 支持报告风格配置（ReportStyle）
- 支持上下文注入

### 5. 健壮的错误处理

- 异常捕获和日志记录
- 回退到原始提示词
- 不中断用户流程

## 扩展性

### 添加新的处理节点

如果需要扩展工作流，可以在 builder.py 中添加新节点：

```python
def build_graph():
    builder = StateGraph(PromptEnhancerState)

    # 添加多个节点
    builder.add_node("validator", validate_prompt_node)
    builder.add_node("enhancer", prompt_enhancer_node)
    builder.add_node("optimizer", optimize_prompt_node)

    # 定义节点流转
    builder.set_entry_point("validator")
    builder.add_edge("validator", "enhancer")
    builder.add_edge("enhancer", "optimizer")
    builder.set_finish_point("optimizer")

    return builder.compile()
```

### 扩展状态字段

在 state.py 中添加新字段：

```python
class PromptEnhancerState(TypedDict):
    prompt: str
    context: Optional[str]
    report_style: Optional[ReportStyle]
    output: Optional[str]

    # 新增字段
    validation_result: Optional[bool]
    optimization_score: Optional[float]
```

### 自定义解析策略

在 enhancer_node.py 中添加新的解析逻辑：

```python
# 策略 3: JSON 格式提取
json_match = re.search(r'\{.*"enhanced_prompt":\s*"(.*?)".*\}', response_content)
if json_match:
    enhanced_prompt = json_match.group(1)
```

## 工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户输入                                  │
│  { prompt, context, report_style, locale }                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  build_graph()                              │
│              创建 LangGraph 工作流                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              prompt_enhancer_node                           │
│  ┌───────────────────────────────────────────────────┐     │
│  │ 1. 获取 LLM 模型                                   │     │
│  │ 2. 构建上下文信息                                  │     │
│  │ 3. 创建 HumanMessage                              │     │
│  │ 4. 应用提示词模板                                  │     │
│  │ 5. 调用 LLM (model.invoke)                        │     │
│  │ 6. 解析响应                                        │     │
│  │    ├─ XML 标签提取 (优先)                         │     │
│  │    └─ 前缀移除回退                                │     │
│  │ 7. 返回增强结果                                    │     │
│  └───────────────────────────────────────────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     输出结果                                  │
│  { prompt, context, report_style, output }                  │
└─────────────────────────────────────────────────────────────┘
```

## 总结

`src/prompt_enhancer` 模块是一个设计精简、功能专注的提示词增强系统，具有以下特点：

**核心优势：**
- ✅ 简洁的单节点工作流设计
- ✅ 基于 LangGraph 的状态管理
- ✅ 双重解析策略（XML + 回退）
- ✅ 国际化支持（多语言模板）
- ✅ 灵活的配置系统
- ✅ 健壮的错误处理和回退机制
- ✅ 详细的日志记录

**适用场景：**
- 自动优化用户输入的提示词
- 提高 AI 响应质量
- 标准化提示词格式
- 支持多语言提示词增强

**代码质量：**
- 清晰的模块划分（state、node、builder）
- 完善的类型注解（TypedDict）
- 详细的文档注释
- 健壮的异常处理
- 完整的单元测试覆盖

**设计哲学：**
- **简单优于复杂**：单节点设计，专注核心功能
- **健壮优于完美**：回退机制确保系统稳定
- **灵活优于固定**：支持多种配置和扩展

该模块为 Deer-Flow 项目提供了可靠、高效的提示词增强能力，是 AI 工作流中的重要组成部分。

---

**文档版本**: 1.0
**最后更新**: 2026-02-11
**维护者**: Deer-Flow Team

