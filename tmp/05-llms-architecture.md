# LLMs 模块架构文档

## 概述

`src/llms` 模块是 Deer-Flow 项目中负责大语言模型（LLM）管理和交互的核心模块。该模块提供了统一的接口来管理多种 LLM 提供商，支持配置管理、实例缓存、流式响应处理等功能。

## 目录结构

```
src/llms/
├── __init__.py              # 模块初始化文件
├── llm.py                   # LLM 管理核心模块
└── providers/
    └── dashscope.py         # Dashscope 提供商实现
```

## 核心模块详解

### 1. llm.py - LLM 管理核心

这是整个 LLMs 模块的核心文件，负责 LLM 实例的创建、配置和管理。

#### 主要功能

**1.1 LLM 类型定义**

支持四种 LLM 类型：
- `reasoning`: 推理模型，用于复杂的推理任务
- `basic`: 基础模型，用于一般对话和文本生成
- `vision`: 视觉模型，用于图像理解任务
- `code`: 代码模型，用于代码生成和理解

**1.2 配置管理**

配置来源优先级：
1. 环境变量（最高优先级）
2. YAML 配置文件（`conf.yaml`）

环境变量格式：`{LLM_TYPE}_MODEL__{KEY}`
- 例如：`BASIC_MODEL__api_key`, `BASIC_MODEL__base_url`

**1.3 支持的 LLM 提供商**

- **OpenAI**: 标准 OpenAI API
- **Azure OpenAI**: 微软 Azure 平台的 OpenAI 服务
- **Google AI Studio**: Google Gemini 模型
- **DeepSeek**: DeepSeek 推理模型
- **Dashscope**: 阿里云通义千问模型

#### 核心函数

##### `get_llm_by_type(llm_type: LLMType) -> BaseChatModel`

获取指定类型的 LLM 实例，支持实例缓存以提高性能。

```python
# 使用示例
reasoning_llm = get_llm_by_type("reasoning")
basic_llm = get_llm_by_type("basic")
```

##### `_create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> BaseChatModel`

根据配置创建 LLM 实例，自动识别提供商类型：

1. **Google AI Studio 识别**：通过 `platform` 字段判断
2. **Azure OpenAI 识别**：通过 `azure_endpoint` 字段判断
3. **Dashscope 识别**：通过 `base_url` 中包含 "dashscope." 判断
4. **DeepSeek 识别**：推理模型且不满足上述条件
5. **默认 OpenAI**：其他情况

##### `get_llm_token_limit_by_type(llm_type: str) -> int`

获取指定 LLM 类型的 token 限制，优先级：
1. 配置文件中显式设置的 `token_limit`
2. 根据模型名称推断的限制
3. 安全默认值（100,000 tokens）

**内置模型 Token 限制：**
- GPT-4o: 120,000
- GPT-4-turbo: 120,000
- Claude-3: 180,000
- Gemini-2: 180,000
- Gemini-1.5-pro/flash: 180,000
- Doubao: 200,000
- DeepSeek: 100,000

##### `get_configured_llm_models() -> dict[str, list[str]]`

获取所有已配置的 LLM 模型，按类型分组。

#### 配置过滤机制

为防止无效参数传递给 LLM 构造函数（Issue #411），模块定义了 `ALLOWED_LLM_CONFIG_KEYS` 白名单：

**通用配置键：**
- `model`, `api_key`, `base_url`, `api_base`
- `max_retries`, `timeout`, `max_tokens`
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`
- `stop`, `n`, `stream`, `logprobs`

**平台特定配置键：**
- Azure: `azure_endpoint`, `azure_deployment`, `api_version`
- Google: `google_api_key`, `platform`
- Dashscope/Doubao: `extra_body`

**SSL 和 HTTP 客户端：**
- `verify_ssl`, `http_client`, `http_async_client`

#### SSL 验证处理

支持禁用 SSL 验证（用于开发环境）：

```python
# 配置示例
BASIC_MODEL:
  model: "gpt-4"
  api_key: "your-key"
  base_url: "https://your-endpoint"
  verify_ssl: false  # 禁用 SSL 验证
```

当 `verify_ssl` 为 `false` 时，会创建自定义的 `httpx.Client` 和 `httpx.AsyncClient`。

### 2. providers/dashscope.py - Dashscope 提供商实现

这个模块扩展了 LangChain 的 `ChatOpenAI` 类，为阿里云通义千问（Dashscope）提供专门的支持，特别是对推理模型的 `reasoning_content` 字段的处理。

#### 核心类：ChatDashscope

继承自 `ChatOpenAI`，增强了对推理内容的支持。

**主要特性：**
1. 支持推理模型的 `reasoning_content` 字段
2. 兼容流式和非流式响应
3. 完整的消息类型转换支持
4. 工具调用（Tool Calls）支持

#### 核心函数

##### `_convert_delta_to_message_chunk(delta_dict, default_class)`

将流式响应的 delta 字典转换为消息块对象。

**支持的消息类型：**
- `HumanMessageChunk`: 用户消息
- `AIMessageChunk`: AI 助手消息（支持 reasoning_content）
- `SystemMessageChunk`: 系统消息
- `FunctionMessageChunk`: 函数调用消息
- `ToolMessageChunk`: 工具调用消息
- `ChatMessageChunk`: 通用聊天消息

**特殊处理：**
- **Function Calls**: 处理函数调用数据，确保 name 字段不为 None
- **Tool Calls**: 转换工具调用为 `tool_call_chunk` 对象
- **Reasoning Content**: 提取并保存推理过程内容（用于推理模型）

##### `_convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)`

将原始流式响应块转换为 `ChatGenerationChunk` 对象。

**处理逻辑：**
1. 跳过 `content.delta` 类型的块（beta API 格式）
2. 提取 token 使用情况（usage metadata）
3. 处理空 choices 的情况
4. 添加完成原因（finish_reason）和模型信息
5. 添加对数概率（logprobs）信息

##### `ChatDashscope._create_chat_result(response, generation_info)`

创建聊天结果对象，从 OpenAI 响应中提取推理内容。

**关键步骤：**
1. 调用父类方法创建基础结果
2. 检查响应是否为 `openai.BaseModel` 类型
3. 提取 `reasoning_content` 字段
4. 将推理内容添加到消息的 `additional_kwargs` 中

##### `ChatDashscope._stream(messages, stop, run_manager, **kwargs)`

创建流式聊天完成生成器。

**流式处理流程：**
1. 设置 `stream=True` 参数
2. 获取请求 payload
3. 根据是否有 `response_format` 选择不同的 API：
   - 有 `response_format`: 使用 `beta.chat.completions.stream`
   - 无 `response_format`: 使用标准 `create` API
4. 逐块处理响应：
   - 转换为字典格式
   - 转换为 generation chunk
   - 更新默认消息类型
   - 触发回调（on_llm_new_token）
5. 处理最终完成（final_completion）

**错误处理：**
- 捕获 `openai.BadRequestError` 并使用 `_handle_openai_bad_request` 处理
- 优雅处理缺失的方法和属性

## 架构设计

### 设计模式

#### 1. 工厂模式

`_create_llm_use_conf` 函数实现了工厂模式，根据配置自动创建合适的 LLM 实例：

```
配置 → 识别提供商 → 创建对应实例
```

**提供商识别逻辑：**
```
if platform == "google_aistudio":
    return ChatGoogleGenerativeAI
elif "azure_endpoint" in config:
    return AzureChatOpenAI
elif "dashscope." in base_url:
    return ChatDashscope
elif llm_type == "reasoning":
    return ChatDeepSeek
else:
    return ChatOpenAI
```

#### 2. 单例模式（缓存机制）

使用 `_llm_cache` 字典缓存 LLM 实例，避免重复创建：

```python
_llm_cache: dict[LLMType, BaseChatModel] = {}

def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm
```

**优势：**
- 减少初始化开销
- 保持连接池复用
- 提高响应速度

#### 3. 适配器模式

`ChatDashscope` 类作为适配器，将 Dashscope API 适配到 LangChain 的标准接口：

```
Dashscope API → ChatDashscope → LangChain Interface
```

### 数据流

#### 非流式调用流程

```
用户请求
  ↓
get_llm_by_type(llm_type)
  ↓
检查缓存 (_llm_cache)
  ↓
[缓存未命中] → _create_llm_use_conf()
  ↓
加载配置 (YAML + 环境变量)
  ↓
过滤配置键 (ALLOWED_LLM_CONFIG_KEYS)
  ↓
识别提供商类型
  ↓
创建 LLM 实例
  ↓
调用 LLM (invoke/ainvoke)
  ↓
_create_chat_result()
  ↓
提取 reasoning_content (如果存在)
  ↓
返回结果
```

#### 流式调用流程

```
用户请求 (stream=True)
  ↓
ChatDashscope._stream()
  ↓
构建请求 payload
  ↓
选择 API (standard/beta)
  ↓
逐块接收响应
  ↓
_convert_chunk_to_generation_chunk()
  ↓
_convert_delta_to_message_chunk()
  ↓
触发回调 (on_llm_new_token)
  ↓
yield ChatGenerationChunk
  ↓
[循环直到完成]
  ↓
处理 final_completion (如果存在)
```

### 配置管理架构

#### 配置层次结构

```
环境变量 (最高优先级)
  ↓
YAML 配置文件
  ↓
默认值 (代码中定义)
```

#### 配置键映射

```
LLM Type          Config Key
---------         -----------
reasoning    →    REASONING_MODEL
basic        →    BASIC_MODEL
vision       →    VISION_MODEL
code         →    CODE_MODEL
```

## 配置示例

### 1. OpenAI 配置

```yaml
BASIC_MODEL:
  model: "gpt-4o"
  api_key: "sk-xxx"
  base_url: "https://api.openai.com/v1"
  max_retries: 3
  temperature: 0.7
  max_tokens: 4096
```

### 2. Azure OpenAI 配置

```yaml
REASONING_MODEL:
  model: "gpt-4"
  api_key: "your-azure-key"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  azure_deployment: "gpt-4-deployment"
  api_version: "2024-02-15-preview"
```

### 3. Google AI Studio 配置

```yaml
VISION_MODEL:
  platform: "google_aistudio"
  model: "gemini-2.0-flash-exp"
  api_key: "your-google-api-key"
  temperature: 0.5
```

### 4. Dashscope (通义千问) 配置

```yaml
REASONING_MODEL:
  model: "qwen-max"
  api_key: "sk-xxx"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.7
  # 推理模型会自动添加 extra_body: {enable_thinking: true}
```

### 5. DeepSeek 配置

```yaml
REASONING_MODEL:
  model: "deepseek-reasoner"
  api_key: "sk-xxx"
  base_url: "https://api.deepseek.com"
  temperature: 1.0
```

### 6. 环境变量配置

```bash
# 基础模型配置
export BASIC_MODEL__model="gpt-4o"
export BASIC_MODEL__api_key="sk-xxx"
export BASIC_MODEL__base_url="https://api.openai.com/v1"

# 推理模型配置
export REASONING_MODEL__model="deepseek-reasoner"
export REASONING_MODEL__api_key="sk-xxx"
export REASONING_MODEL__base_url="https://api.deepseek.com"
```

### 7. 禁用 SSL 验证（开发环境）

```yaml
BASIC_MODEL:
  model: "gpt-4"
  api_key: "sk-xxx"
  base_url: "https://your-local-endpoint"
  verify_ssl: false  # 仅用于开发环境
```

### 8. Token 限制配置

```yaml
BASIC_MODEL:
  model: "gpt-4o"
  api_key: "sk-xxx"
  token_limit: 120000  # 显式设置 token 限制
```

## 使用示例

### 1. 基础使用

```python
from src.llms.llm import get_llm_by_type

# 获取基础模型
basic_llm = get_llm_by_type("basic")

# 同步调用
response = basic_llm.invoke("你好，请介绍一下自己")
print(response.content)

# 异步调用
response = await basic_llm.ainvoke("你好，请介绍一下自己")
print(response.content)
```

### 2. 流式响应

```python
from src.llms.llm import get_llm_by_type

reasoning_llm = get_llm_by_type("reasoning")

# 流式调用
for chunk in reasoning_llm.stream("解释量子纠缠的原理"):
    print(chunk.content, end="", flush=True)

# 异步流式调用
async for chunk in reasoning_llm.astream("解释量子纠缠的原理"):
    print(chunk.content, end="", flush=True)
```

### 3. 获取推理内容（Reasoning Content）

```python
from src.llms.llm import get_llm_by_type

reasoning_llm = get_llm_by_type("reasoning")
response = reasoning_llm.invoke("计算 123 * 456 的结果")

# 获取推理过程
reasoning_content = response.additional_kwargs.get("reasoning_content")
if reasoning_content:
    print("推理过程:", reasoning_content)

# 获取最终答案
print("答案:", response.content)
```

### 4. 使用工具调用（Tool Calls）

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}的天气是晴天"

llm = get_llm_by_type("basic")
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("北京的天气怎么样？")
print(response.tool_calls)
```

### 5. 查询已配置的模型

```python
from src.llms.llm import get_configured_llm_models

models = get_configured_llm_models()
print(models)
# 输出: {'basic': ['gpt-4o'], 'reasoning': ['deepseek-reasoner'], ...}
```

### 6. 获取 Token 限制

```python
from src.llms.llm import get_llm_token_limit_by_type

# 获取基础模型的 token 限制
limit = get_llm_token_limit_by_type("basic")
print(f"基础模型 token 限制: {limit}")
```

## 最佳实践

### 1. 配置管理

**推荐做法：**
- 敏感信息（API Key）使用环境变量存储
- 非敏感配置（model, temperature）使用 YAML 文件
- 生产环境启用 SSL 验证（`verify_ssl: true`）

```bash
# .env 文件
BASIC_MODEL__api_key=sk-xxx
REASONING_MODEL__api_key=sk-xxx
```

```yaml
# conf.yaml
BASIC_MODEL:
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 4096
```

### 2. 错误处理

```python
from src.llms.llm import get_llm_by_type
from langchain_core.exceptions import LangChainException

try:
    llm = get_llm_by_type("basic")
    response = llm.invoke("你好")
except ValueError as e:
    print(f"配置错误: {e}")
except LangChainException as e:
    print(f"LLM 调用错误: {e}")
```

### 3. Token 管理

```python
from src.llms.llm import get_llm_token_limit_by_type

# 在发送请求前检查 token 限制
token_limit = get_llm_token_limit_by_type("basic")
estimated_tokens = len(prompt) // 4  # 粗略估算

if estimated_tokens > token_limit * 0.8:  # 留 20% 余量
    print("警告: 输入可能超过 token 限制")
```

### 4. 缓存利用

```python
# 推荐：复用 LLM 实例
llm = get_llm_by_type("basic")
for query in queries:
    response = llm.invoke(query)

# 不推荐：每次都创建新实例
for query in queries:
    llm = get_llm_by_type("basic")  # 会从缓存获取，但不必要
    response = llm.invoke(query)
```

### 5. 异步调用

对于高并发场景，使用异步调用：

```python
import asyncio
from src.llms.llm import get_llm_by_type

async def process_queries(queries):
    llm = get_llm_by_type("basic")
    tasks = [llm.ainvoke(query) for query in queries]
    return await asyncio.gather(*tasks)

# 并发处理多个查询
results = asyncio.run(process_queries(["问题1", "问题2", "问题3"]))
```

## 注意事项

### 1. 配置键过滤

模块会自动过滤不在 `ALLOWED_LLM_CONFIG_KEYS` 白名单中的配置键，并输出警告日志。如果看到类似警告：

```
WARNING: Removed unexpected LLM configuration key 'SEARCH_ENGINE'
```

这表示配置文件中有不属于 LLM 的配置项，应该移到正确的配置节。

### 2. Dashscope 推理模型

使用 Dashscope 的推理模型时，系统会自动添加 `extra_body: {enable_thinking: true}`：

```python
# 自动处理
if "dashscope." in base_url and llm_type == "reasoning":
    merged_conf["extra_body"] = {"enable_thinking": True}
```

### 3. Token 限制推断

如果未显式配置 `token_limit`，系统会根据模型名称自动推断：

- 优先级 1: 配置文件中的 `token_limit`
- 优先级 2: 根据模型名称推断
- 优先级 3: 默认值 100,000

### 4. SSL 验证

生产环境务必启用 SSL 验证（默认启用）。仅在开发环境或内网环境可以考虑禁用。

### 5. 环境变量优先级

环境变量的优先级高于 YAML 配置，可用于：
- 在不同环境使用不同配置
- 保护敏感信息（API Key）
- CI/CD 流程中动态配置

## 常见问题

### Q1: 如何切换不同的 LLM 提供商？

**A:** 修改配置文件中的 `base_url` 和相关参数即可：

```yaml
# 从 OpenAI 切换到 Dashscope
BASIC_MODEL:
  model: "qwen-max"
  api_key: "your-dashscope-key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### Q2: 为什么会出现 "Removed unexpected LLM configuration key" 警告？

**A:** 这是因为配置文件中包含了不属于 LLM 的配置项。检查配置文件，将非 LLM 配置移到正确的位置。

### Q3: 如何使用本地部署的模型？

**A:** 配置 `base_url` 指向本地服务地址：

```yaml
BASIC_MODEL:
  model: "llama-3"
  api_key: "not-needed"
  base_url: "http://localhost:8000/v1"
```

### Q4: 推理内容（reasoning_content）在哪里？

**A:** 推理内容存储在响应的 `additional_kwargs` 中：

```python
response = llm.invoke("问题")
reasoning = response.additional_kwargs.get("reasoning_content")
```

### Q5: 如何处理 token 超限错误？

**A:**
1. 检查当前 token 限制：`get_llm_token_limit_by_type("basic")`
2. 在配置中显式设置更大的限制（如果模型支持）
3. 压缩输入内容或使用上下文压缩技术

### Q6: 支持哪些 LangChain 功能？

**A:** 所有返回的 LLM 实例都是标准的 `BaseChatModel`，支持：
- 同步/异步调用（invoke/ainvoke）
- 流式响应（stream/astream）
- 工具调用（bind_tools）
- 批处理（batch/abatch）
- LangChain 表达式语言（LCEL）

## 技术亮点

### 1. 统一接口设计

通过工厂模式和适配器模式，为不同的 LLM 提供商提供统一的调用接口，降低切换成本。

### 2. 智能配置管理

- 多层配置优先级（环境变量 > YAML > 默认值）
- 自动配置键过滤，防止无效参数传递
- 智能 token 限制推断

### 3. 性能优化

- LLM 实例缓存机制
- HTTP 连接池复用
- 支持异步并发调用

### 4. 推理能力增强

`ChatDashscope` 类专门处理推理模型的 `reasoning_content`，完整保留模型的思考过程。

### 5. 错误处理

- 优雅的错误处理和降级
- 详细的日志和警告信息
- 自动重试机制（max_retries）

## 扩展性

### 添加新的 LLM 提供商

1. 在 `_create_llm_use_conf` 中添加识别逻辑
2. 如需特殊处理，创建新的适配器类（参考 `ChatDashscope`）
3. 更新 `ALLOWED_LLM_CONFIG_KEYS` 添加特定配置键

### 添加新的 LLM 类型

1. 在 `LLMType` 中添加新类型
2. 在 `_get_llm_type_config_keys` 中添加映射
3. 在配置文件中添加对应配置节

## 总结

`src/llms` 模块是一个设计良好、功能完善的 LLM 管理系统，具有以下特点：

**核心优势：**
- ✅ 统一的多提供商支持（OpenAI、Azure、Google、DeepSeek、Dashscope）
- ✅ 灵活的配置管理（YAML + 环境变量）
- ✅ 高性能缓存机制
- ✅ 完整的流式响应支持
- ✅ 推理内容（reasoning_content）处理
- ✅ 智能 token 限制管理
- ✅ 良好的错误处理和日志

**适用场景：**
- 需要支持多个 LLM 提供商的应用
- 需要在不同环境使用不同配置的项目
- 需要推理能力的复杂任务
- 高并发的 LLM 调用场景

**代码质量：**
- 清晰的模块划分
- 完善的类型注解
- 详细的文档注释
- 健壮的错误处理

该模块为 Deer-Flow 项目提供了稳定、高效、易扩展的 LLM 接入能力。

---

**文档版本**: 1.0
**最后更新**: 2026-02-11
**维护者**: Deer-Flow Team


