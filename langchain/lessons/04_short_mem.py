#!/usr/bin/env python3
"""
04. 短期记忆代理 - 对话历史管理

学习目标：
- 理解短期记忆在对话系统中的作用
- 掌握如何在 LangChain 中管理对话历史
- 学习状态保持和上下文感知

Author: LangChain Learning
Date: 2026-01-25
"""

# 导入部分
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

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
    # 在实际应用中，这里会存储到数据库或内存中
    return f"已记住：{fact}"

# 初始化模型
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1
)

# 创建支持记忆的代理
agent = create_agent(
    model=model,
    tools=[get_weather, calculate, remember_fact],
    system_prompt="""你是一个智能助手，具有短期记忆能力。

记忆管理规则：
1. 记住用户的重要信息和偏好
2. 在对话中引用之前的上下文
3. 主动提醒用户之前提到的事情
4. 保持对话的连贯性

请在回答时体现出你记住了之前的对话内容。"""
)

# 模拟多轮对话，展示记忆功能
def demonstrate_memory():
    """演示短期记忆功能"""

    # 初始化对话历史
    conversation_history = []

    # 第一轮对话
    print("=== 第一轮对话 ===")
    user_input1 = "你好，我叫小明，来自北京。请记住我喜欢蓝色。"
    print(f"用户: {user_input1}")

    result1 = agent.invoke({
        "messages": conversation_history + [HumanMessage(content=user_input1)]
    })

    ai_response1 = result1["messages"][-1].content
    print(f"助手: {ai_response1}")

    # 更新对话历史
    conversation_history.extend([
        HumanMessage(content=user_input1),
        AIMessage(content=ai_response1)
    ])

    # 第二轮对话
    print("\n=== 第二轮对话 ===")
    user_input2 = "今天北京天气怎么样？"
    print(f"用户: {user_input2}")

    result2 = agent.invoke({
        "messages": conversation_history + [HumanMessage(content=user_input2)]
    })

    ai_response2 = result2["messages"][-1].content
    print(f"助手: {ai_response2}")

    # 更新对话历史
    conversation_history.extend([
        HumanMessage(content=user_input2),
        AIMessage(content=ai_response2)
    ])

    # 第三轮对话
    print("\n=== 第三轮对话 ===")
    user_input3 = "帮我算一下 15 + 27 等于多少？"
    print(f"用户: {user_input3}")

    result3 = agent.invoke({
        "messages": conversation_history + [HumanMessage(content=user_input3)]
    })

    ai_response3 = result3["messages"][-1].content
    print(f"助手: {ai_response3}")

    # 更新对话历史
    conversation_history.extend([
        HumanMessage(content=user_input3),
        AIMessage(content=ai_response3)
    ])

    # 第四轮对话 - 测试记忆
    print("\n=== 第四轮对话（测试记忆）===")
    user_input4 = "你还记得我叫什么名字吗？还有我的喜好是什么？"
    print(f"用户: {user_input4}")

    result4 = agent.invoke({
        "messages": conversation_history + [HumanMessage(content=user_input4)]
    })

    ai_response4 = result4["messages"][-1].content
    print(f"助手: {ai_response4}")

# 执行演示
if __name__ == "__main__":
    demonstrate_memory()