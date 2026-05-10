#!/usr/bin/env python3
"""
04. 短期记忆代理 - 使用 Checkpoint 管理状态

学习目标：
- 理解 LangChain 中 checkpoint 的工作原理
- 掌握如何使用 MemorySaver 实现状态持久化
- 学习真正的记忆管理，而非传递整个历史

Author: LangChain Learning
Date: 2026-01-25
"""

# 导入部分
import middleware_debugging

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴天，温度 5°C",
        "上海": "多云，温度 8°C",
        "广州": "阴天，温度 15°C",
        "深圳": "小雨，温度 12°C"
    }
    return weather_data.get(city, f"{city} 天气信息暂未收录")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def remember_fact(fact: str) -> str:
    """记住一个重要事实"""
    return f"已记住重要信息：{fact}"

# 初始化模型
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1
)

# 创建记忆检查点
memory = MemorySaver()


logging_middleware = middleware_debugging.DetailedLoggingMiddleware(log_level="DEBUG")
performance_middleware = middleware_debugging.PerformanceMonitoringMiddleware()

# 创建支持 checkpoint 的代理
agent = create_agent(
    model=model,
    tools=[get_weather, calculate, remember_fact],
    middleware=[logging_middleware, performance_middleware],
    checkpointer=memory,  # 使用 checkpoint 管理状态
    system_prompt="""你是一个智能助手，具有持久记忆能力。

记忆管理规则：
1. 记住用户的重要信息和偏好
2. 在对话中引用之前的上下文
3. 主动提醒用户之前提到的事情
4. 保持对话的连贯性

通过 checkpoint 机制，你的记忆会被自动保存和恢复。"""
)

# 模拟多轮对话，展示 checkpoint 记忆功能
def demonstrate_checkpoint_memory():
    """演示使用 checkpoint 的记忆功能"""

    # 使用固定的 thread_id 来保持会话连续性
    config = {"configurable": {"thread_id": "user_session_001"}}

    # 第一轮对话
    print("=== 第一轮对话 ===")
    user_input1 = "你好，我叫小明，来自北京。请记住我喜欢蓝色。"
    print(f"用户: {user_input1}")

    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": user_input1}]},
        config=config
    )

    ai_response1 = result1["messages"][-1].content
    print(f"助手: {ai_response1}")

    # 显示第一轮的详细执行信息
    print(f"📝 消息历史长度: {len(result1['messages'])}")
    tool_messages = [msg for msg in result1['messages'] if msg.type == 'tool']
    if tool_messages:
        print(f"🔧 工具调用: {len(tool_messages)} 次")
        for msg in tool_messages:
            print(f"  - 工具结果: {msg.content[:50]}...")
    print()
    print("\n=== 第二轮对话 ===")
    user_input2 = "今天北京天气怎么样？"
    print(f"用户: {user_input2}")

    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": user_input2}]},
        config=config  # 使用相同的 thread_id
    )

    ai_response2 = result2["messages"][-1].content
    print(f"助手: {ai_response2}")

    # 显示第二轮的详细执行信息
    print(f"📝 消息历史长度: {len(result2['messages'])}")
    tool_messages = [msg for msg in result2['messages'] if msg.type == 'tool']
    if tool_messages:
        print(f"🔧 工具调用: {len(tool_messages)} 次")
        for msg in tool_messages:
            print(f"  - 工具: {msg.name}, 结果: {msg.content[:50]}...")
    print()
    print("\n=== 第三轮对话 ===")
    user_input3 = "帮我算一下 15 + 27 等于多少？"
    print(f"用户: {user_input3}")

    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": user_input3}]},
        config=config
    )

    ai_response3 = result3["messages"][-1].content
    print(f"助手: {ai_response3}")

    # 显示第三轮的详细执行信息
    print(f"📝 消息历史长度: {len(result3['messages'])}")
    tool_messages = [msg for msg in result3['messages'] if msg.type == 'tool']
    if tool_messages:
        print(f"🔧 工具调用: {len(tool_messages)} 次")
        for msg in tool_messages:
            print(f"  - 工具: {msg.name}, 结果: {msg.content[:50]}...")
    print()

    # 第四轮对话 - 测试记忆
    user_input4 = "你还记得我叫什么名字吗？还有我的喜好是什么？"
    print(f"用户: {user_input4}")

    result4 = agent.invoke(
        {"messages": [{"role": "user", "content": user_input4}]},
        config=config
    )

    ai_response4 = result4["messages"][-1].content
    print(f"助手: {ai_response4}")

    # 显示第四轮的详细执行信息
    print(f"📝 消息历史长度: {len(result4['messages'])}")
    tool_messages = [msg for msg in result4['messages'] if msg.type == 'tool']
    if tool_messages:
        print(f"🔧 工具调用: {len(tool_messages)} 次")
        for msg in tool_messages:
            print(f"  - 工具: {msg.name}, 结果: {msg.content[:50]}...")
    print()
    print("\n=== 跨会话恢复测试 ===")
    print("模拟重新启动应用，使用相同的 thread_id...")

    # 创建新的代理实例（模拟应用重启）
    new_agent = create_agent(
        model=model,
        tools=[get_weather, calculate, remember_fact],
        checkpointer=memory,  # 使用相同的 memory 实例
        system_prompt="你是一个智能助手，具有持久记忆能力。"
    )

    user_input5 = "我之前告诉过你什么关于我的信息吗？"
    print(f"用户: {user_input5}")

    result5 = new_agent.invoke(
        {"messages": [{"role": "user", "content": user_input5}]},
        config=config  # 使用相同的 thread_id
    )

    ai_response5 = result5["messages"][-1].content
    print(f"助手: {ai_response5}")

    # 显示第五轮的详细执行信息
    print(f"📝 消息历史长度: {len(result5['messages'])}")
    tool_messages = [msg for msg in result5['messages'] if msg.type == 'tool']
    if tool_messages:
        print(f"🔧 工具调用: {len(tool_messages)} 次")
        for msg in tool_messages:
            print(f"  - 工具: {msg.name}, 结果: {msg.content[:50]}...")
    print()

# 执行演示
if __name__ == "__main__":
    demonstrate_checkpoint_memory()