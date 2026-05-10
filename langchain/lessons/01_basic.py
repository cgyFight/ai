#!/usr/bin/env python3
"""
01. 基础概念 - 使用千问模型的智能代理

学习目标：
- 理解 LangChain 代理的基本概念
- 掌握如何配置和使用千问模型
- 学习环境变量管理

Author: LangChain Learning
Date: 2026-01-24
"""

# 导入部分
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 工具函数
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city} 天气晴朗，温度 25°C"

# 初始化千问模型
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1
)

# 创建代理
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个有用的天气助手"
)

# 运行代理
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    
    # 提取最终的模型回答
    final_message = result["messages"][-1]
    print(final_message.content)