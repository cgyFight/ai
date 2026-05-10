# LangChain 学习项目 - 代理开发指南

## 项目概述

这是一个 LangChain 学习项目，专注于通过实践掌握 LangChain 框架的核心概念。项目采用渐进式学习方法，每个模块专注于特定的功能领域。

### 项目特点
- **教育导向**：代码设计优先考虑学习价值
- **渐进式难度**：从基础概念到高级应用
- **实用性强**：所有代码示例均可直接运行
- **中文友好**：支持中文注释和错误信息

## 构建和运行命令

### 环境管理
```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv sync

# 添加新依赖
uv add package_name

# 运行单个文件
uv run python 文件名.py

# 运行主入口
uv run python main.py
```

### 单个文件测试
```bash
# 运行基础示例
uv run python 01_basic.py

# 运行特定学习模块
uv run python lessons/02_chains.py

# 验证环境配置
uv run python test_setup.py
```

### 依赖管理
```bash
# 查看已安装依赖
uv pip list

# 更新依赖
uv sync --upgrade

# 添加开发依赖
uv add --optional dev package_name
```

## 代码风格指南

### 导入规范
```python
# 标准库导入
import os
from typing import List, Dict, Optional

# LangChain 核心模块
from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# 第三方库
from dotenv import load_dotenv
```

### 类型提示要求
```python
# 函数必须有类型提示
def process_text(input_text: str, max_length: int = 100) -> Optional[str]:
    """处理输入文本的函数"""
    pass

# 使用 TypedDict 定义结构化数据
class AgentState(TypedDict):
    messages: List[str]
    current_step: str
```

### 命名约定
```python
# 变量和函数：snake_case
weather_data = {}
current_temperature = 25
def get_weather_info(city: str) -> Dict[str, str]:

# 类名：PascalCase
class WeatherAgent:
    def __init__(self, api_key: str):

# 常量：UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_MODEL = "gpt-3.5-turbo"
```

### 错误处理标准
```python
# 提供清晰的中文错误信息
try:
    response = agent.invoke({"messages": [HumanMessage(content=message)]})
    return response.content
except ValueError as e:
    raise ValueError(f"输入参数错误: {e}")
except ConnectionError as e:
    raise ConnectionError(f"API 连接失败，请检查网络设置: {e}")
except Exception as e:
    raise RuntimeError(f"处理过程中发生未知错误: {e}")
```

### 文档字符串格式
```python
def create_weather_agent(model_name: str, tools: List[callable]) -> AgentExecutor:
    """创建天气查询代理。
    
    Args:
        model_name: 要使用的语言模型名称
        tools: 代理可使用的工具列表
        
    Returns:
        配置好的 AgentExecutor 实例
        
    Raises:
        ValueError: 当模型名称无效时
        ConnectionError: 当无法连接到 API 时
    """
    pass
```

## 文件组织规范

### 目录结构
```
langchain-gent/
├── lessons/           # 学习模块
│   ├── 01_basic.py   # 基础代理 - 使用千问模型
│   ├── 02_react.py   # ReAct 代理 - 推理与行动模式
│   ├── 03_plan_and_execute.py  # Plan-and-Execute 代理 - 规划与执行模式
│   ├── 04_short_mem.py    # 短期记忆 - 手动历史传递
│   └── 04_short_mem2.py   # 短期记忆 - 使用 Checkpoint
├── projects/         # 实践项目
│   ├── chatbot.py    # 聊天机器人
│   ├── rag_system.py # RAG 系统
│   └── agent_tools.py # 代理工具
├── utils/           # 工具函数
│   ├── config.py    # 配置管理
│   └── helpers.py   # 辅助函数
└── tests/           # 简单测试
    └── test_setup.py
```

### 文件命名规范
- 学习模块：`XX_模块名.py`（如 `01_basic.py`）
- 实践项目：`项目名.py`（如 `chatbot.py`）
- 工具函数：`功能名_helpers.py`（如 `config_helpers.py`）

### 单文件结构模板
```python
#!/usr/bin/env python3
"""
XX. 模块名称 - 简要描述

学习目标：
- 理解核心概念
- 掌握基本用法
- 能够解决实际问题

Author: LangChain Learning
Date: 2025-01-24
"""

# 导入部分
import os
from typing import List, Dict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 核心功能实现
def main():
    """主要功能演示"""
    pass

if __name__ == "__main__":
    main()
```

## LangChain 特定模式

### Agent 创建标准模式
```python
# 1. 定义工具函数
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city} 天气晴朗，温度 25°C"

# 2. 初始化模型
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 3. 创建 Agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个有用的天气助手"
)

# 4. 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_weather],
    verbose=True
)
```

### 环境变量管理
```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "langchain-gent")
    
    @classmethod
    def validate_keys(cls) -> None:
        """验证必需的 API 密钥是否存在"""
        required_keys = ["OPENAI_API_KEY"]
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            raise ValueError(f"缺少必需的环境变量: {', '.join(missing_keys)}")
```

### Chain 创建模式
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建提示模板
prompt = ChatPromptTemplate.from_template(
    "请解释 {topic}，使用 {language} 语言"
)

# 创建链
chain = prompt | model | StrOutputParser()

# 执行链
result = chain.invoke({
    "topic": "LangChain",
    "language": "中文"
})
```

### 输出解析器使用
```python
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# JSON 输出解析
json_parser = JsonOutputParser()
json_chain = prompt | model | json_parser

# 结构化输出
structured_output = model.with_structured_output({
    "temperature": "float",
    "description": "string",
    "city": "string"
})
```

## 最佳实践

### 1. 代码可读性
- 使用有意义的变量名和函数名
- 添加充分的中文注释
- 每个函数只做一件事
- 保持代码简洁明了

### 2. 错误处理
- 提供清晰的中文错误信息
- 使用具体的异常类型
- 包含错误恢复建议
- 记录重要操作日志

### 5. 代理模式选择
- **ReAct 模式**：适合探索性任务，需要灵活推理和行动交替
- **Plan-and-Execute 模式**：适合结构化任务，有明确执行步骤
- **根据任务复杂度选择合适的模式**

### 6. 记忆管理策略
- **短期记忆**：使用 `checkpointer` 实现状态持久化，避免传递整个对话历史
- **Thread ID**：通过 `configurable: {"thread_id": "session_id"}` 管理独立会话
- **跨会话恢复**：相同 `thread_id` 可恢复之前的状态

### 7. API 密钥安全
- **环境变量**：使用 `.env` 文件和 `load_dotenv()` 管理密钥
- **密钥验证**：实现 `validate_keys()` 函数检查必需的 API 密钥
- **权限控制**：根据最小权限原则配置 API 访问权限

### 8. 模型配置优化
- **温度设置**：推理任务使用较低温度（0.1），创造性任务使用较高温度
- **超时配置**：设置合理的 `timeout` 避免长时间等待
- **兼容接口**：使用 OpenAI 兼容接口集成不同模型提供商

### 9. 工具设计原则
- **单一职责**：每个工具只负责一个明确的功能
- **错误处理**：工具内部实现异常处理，返回用户友好的错误信息
- **参数验证**：使用类型提示和参数校验确保输入有效性

### 10. 性能和扩展性
- **批量处理**：对于多个相似任务，考虑批量处理优化
- **缓存机制**：使用 `cache` 参数启用结果缓存
- **异步支持**：对于长时间任务，考虑异步处理方式

## 高级代理模式

### ReAct (Reasoning + Acting) 模式
```python
# 特点：推理和行动交替进行
agent = create_agent(
    model=model,
    tools=[search_tool, analyze_tool],
    system_prompt="你是一个研究助手，通过推理和行动来解决问题。"
)
# 适用场景：探索性任务、需要灵活调整的复杂问题
```

### Plan-and-Execute 模式
```python
# 特点：先制定完整计划，再按步骤执行
agent = create_agent(
    model=model,
    tools=[research_tool, analyze_tool, report_tool],
    system_prompt="""你是一个项目经理，使用 Plan-and-Execute 模式工作。
    
工作流程：
1. 首先制定详细的执行计划
2. 按计划顺序执行每个步骤
3. 只有在所有步骤完成后才给出最终答案"""
)
# 适用场景：结构化任务、多步骤工作流
```

### 记忆管理实现
```python
# 使用 Checkpoint 实现真正的状态持久化
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory,  # 关键参数
    system_prompt="你具有持久记忆能力"
)

# 会话管理
config = {"configurable": {"thread_id": "user_session_001"}}
result = agent.invoke({"messages": [user_message]}, config=config)
```

## 常见问题解决

### 1. ImportError 问题
```python
# 错误：create_react_agent 不存在
# 解决：使用 create_agent 替代
from langchain.agents import create_agent  # 而不是 create_react_agent
```

### 2. 模型连接问题
```python
# 错误：API 连接超时
# 解决：添加超时设置
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    timeout=30  # 添加超时
)
```

### 3. 记忆状态丢失
```python
# 错误：每次对话都忘记之前内容
# 解决：使用相同的 thread_id
config = {"configurable": {"thread_id": "unique_session_id"}}
# 每次调用都使用相同的 config
```

### 4. 工具执行失败
```python
# 错误：工具返回异常
# 解决：在工具函数中添加异常处理
@tool
def search_function(query: str) -> str:
    try:
        # 工具实现
        return result
    except Exception as e:
        return f"搜索失败: {e}"
```

## 扩展方向

### 1. 多模型集成
- 支持同时使用多个不同的模型
- 根据任务类型自动选择最适合的模型
- 实现模型切换和负载均衡

### 2. 高级记忆系统
- **长期记忆**：使用数据库存储用户偏好和历史
- **记忆压缩**：智能总结和压缩对话历史
- **跨会话学习**：从历史对话中学习用户模式

### 3. 工具生态系统
- **工具市场**：创建可复用的工具库
- **动态工具加载**：运行时根据需要加载工具
- **工具组合**：支持工具之间的协作和数据传递

### 4. 监控和调试
- **执行追踪**：记录代理的推理过程和决策
- **性能监控**：监控响应时间和资源使用
- **错误分析**：收集和分析失败案例

### 5. 安全和合规
- **内容过滤**：防止生成有害内容
- **隐私保护**：确保用户数据安全
- **审计日志**：记录所有操作以便审计

## 测试和验证

### 简单测试模式
```python
def test_basic_functionality():
    """测试基本功能"""
    try:
        # 测试模型连接
        model = ChatOpenAI()
        response = model.invoke("测试")
        assert "success" in response.content.lower()
        print("✅ 基础功能测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_basic_functionality()
```

### 验证脚本
创建简单的验证脚本来检查：
- 环境变量配置
- API 连接状态
- 依赖包版本
- 基本功能可用性

---

**注意**：此指南专为 LangChain 学习项目设计，优先考虑教育价值而非生产环境的最优化。所有代码示例都应该易于理解、修改和扩展。