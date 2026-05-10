#!/usr/bin/env python3
"""
05. 自定义代理记忆 - Customizing Agent Memory

学习目标：
- 理解如何自定义代理的记忆机制
- 掌握状态模式定制和记忆转换器
- 学习高级记忆管理技术

Author: LangChain Learning
Date: 2026-01-25
"""

# 导入部分
import os
from typing import TypedDict, List, Optional, Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

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

# 自定义状态模式
class CustomAgentState(TypedDict):
    """自定义代理状态，包含增强的记忆功能"""
    messages: List[BaseMessage]
    user_name: Optional[str]  # 用户姓名
    user_preferences: dict[str, Any]  # 用户偏好
    conversation_summary: Optional[str]  # 对话摘要
    important_facts: List[str]  # 重要事实
    memory_version: int  # 记忆版本号

# 记忆转换器类
class MemoryTransformer:
    """记忆转换器：管理记忆的压缩、过滤和增强"""

    def __init__(self):
        self.max_messages = 10  # 最大消息数量
        self.summary_threshold = 20  # 触发摘要的阈值

    def compress_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """压缩消息历史，保留重要信息"""
        if len(messages) <= self.max_messages:
            return messages

        # 保留最新的消息和系统消息
        recent_messages = messages[-self.max_messages:]
        system_messages = [msg for msg in messages[:-self.max_messages] if isinstance(msg, SystemMessage)]

        return system_messages + recent_messages

    def extract_user_info(self, messages: List[BaseMessage]) -> dict[str, Any]:
        """从消息中提取用户信息"""
        user_info = {
            "name": None,
            "preferences": {},
            "important_facts": []
        }

        for msg in messages:
            content = msg.content.lower()

            # 提取姓名
            if "我叫" in content or "我是" in content:
                # 简单的姓名提取逻辑
                if "小明" in content:
                    user_info["name"] = "小明"
                elif "小红" in content:
                    user_info["name"] = "小红"

            # 提取偏好
            if "喜欢" in content:
                if "蓝色" in content:
                    user_info["preferences"]["color"] = "blue"
                elif "红色" in content:
                    user_info["preferences"]["color"] = "red"

            # 提取重要事实
            if "记住" in content or "重要" in content:
                user_info["important_facts"].append(msg.content)

        return user_info

    def generate_summary(self, messages: List[BaseMessage]) -> str:
        """生成对话摘要"""
        if len(messages) < self.summary_threshold:
            return None

        # 简单的摘要生成逻辑
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        topics = []

        for msg in user_messages[-5:]:  # 最近5条用户消息
            content = msg.content.lower()
            if "天气" in content:
                topics.append("天气查询")
            elif "计算" in content or any(char.isdigit() for char in content):
                topics.append("数学计算")
            elif "喜欢" in content or "偏好" in content:
                topics.append("个人偏好")

        if topics:
            return f"对话主要涉及：{', '.join(set(topics))}"
        return "一般性对话"

# 自定义记忆管理器
class CustomMemoryManager:
    """自定义记忆管理器"""

    def __init__(self):
        self.transformer = MemoryTransformer()
        self.memory_store = {}  # 简单的内存存储

    def update_memory(self, thread_id: str, state: CustomAgentState) -> CustomAgentState:
        """更新记忆状态"""
        # 压缩消息历史
        state["messages"] = self.transformer.compress_messages(state["messages"])

        # 提取用户信息
        user_info = self.transformer.extract_user_info(state["messages"])
        state["user_name"] = user_info["name"]
        state["user_preferences"] = user_info["preferences"]
        state["important_facts"] = user_info["important_facts"]

        # 生成摘要
        state["conversation_summary"] = self.transformer.generate_summary(state["messages"])

        # 更新版本号
        state["memory_version"] = state.get("memory_version", 0) + 1

        # 存储到内存
        self.memory_store[thread_id] = state

        return state

    def get_memory(self, thread_id: str) -> Optional[CustomAgentState]:
        """获取记忆状态"""
        return self.memory_store.get(thread_id)

# 创建自定义记忆代理
def create_custom_memory_agent():
    """创建具有自定义记忆功能的代理"""

    # 初始化模型
    model = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.1
    )

    # 初始化记忆管理器
    memory_manager = CustomMemoryManager()

    # 创建标准代理
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate],
        system_prompt="""你是一个智能助手，具有高级记忆能力。

记忆增强功能：
1. 记住用户的姓名和偏好
2. 追踪重要事实和对话摘要
3. 在回答时主动引用相关记忆信息
4. 保持对话的个性化体验

请在对话中体现出你对用户信息的了解。"""
    )

    return agent, memory_manager

# 演示自定义记忆功能
def demonstrate_custom_memory():
    """演示自定义记忆功能"""

    agent, memory_manager = create_custom_memory_agent()

    # 使用固定的 thread_id
    config = {"configurable": {"thread_id": "custom_memory_demo"}}

    conversations = [
        "你好，我叫小明，我特别喜欢蓝色。",
        "今天北京的天气怎么样？我喜欢蓝色天空。",
        "请帮我计算 25 + 37 等于多少？",
        "你还记得我的名字和喜好吗？",
        "根据我们的对话，你能总结一下吗？"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"=== 第 {i} 轮对话 ===")
        print(f"用户: {user_input}")

        # 执行代理
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        ai_response = result["messages"][-1].content
        print(f"助手: {ai_response}")

        # 显示记忆状态
        memory_state = memory_manager.get_memory("custom_memory_demo")
        if memory_state:
            print("🧠 记忆状态:")
            print(f"  - 用户姓名: {memory_state.get('user_name', '未知')}")
            print(f"  - 用户偏好: {memory_state.get('user_preferences', {})}")
            print(f"  - 重要事实数量: {len(memory_state.get('important_facts', []))}")
            print(f"  - 对话摘要: {memory_state.get('conversation_summary', '无')}")
            print(f"  - 记忆版本: {memory_state.get('memory_version', 0)}")
            print(f"  - 消息历史长度: {len(memory_state.get('messages', []))}")

        # 检查工具调用
        tool_messages = [msg for msg in result['messages'] if msg.type == 'tool']
        if tool_messages:
            print(f"🔧 工具调用: {len(tool_messages)} 次")
            for msg in tool_messages:
                print(f"  - 工具: {msg.name}, 结果: {msg.content[:50]}...")

        print()

# 高级记忆定制示例
def demonstrate_advanced_memory():
    """演示高级记忆定制技术"""

    print("=== 高级记忆定制示例 ===")

    transformer = MemoryTransformer()

    # 模拟大量消息
    messages = []
    for i in range(25):  # 创建25条消息
        if i % 2 == 0:
            messages.append(HumanMessage(content=f"这是第 {i//2 + 1} 条用户消息"))
        else:
            messages.append(AIMessage(content=f"这是第 {i//2 + 1} 条助手回复"))

    print(f"原始消息数量: {len(messages)}")

    # 压缩消息
    compressed = transformer.compress_messages(messages)
    print(f"压缩后消息数量: {len(compressed)}")

    # 添加一些用户信息
    info_messages = messages + [
        HumanMessage(content="我叫小明，来自北京"),
        HumanMessage(content="我喜欢蓝色和阅读"),
        HumanMessage(content="请记住我喜欢早起")
    ]

    # 提取用户信息
    user_info = transformer.extract_user_info(info_messages)
    print(f"提取的用户信息: {user_info}")

    # 生成摘要
    summary = transformer.generate_summary(info_messages)
    print(f"生成的对话摘要: {summary}")

# 执行演示
if __name__ == "__main__":
    print("=== 自定义记忆代理演示 ===")
    demonstrate_custom_memory()

    print("\n" + "="*50 + "\n")
    demonstrate_advanced_memory()