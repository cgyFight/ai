#!/usr/bin/env python3
"""
02. ReAct 代理 - 推理与行动代理

学习目标：
- 理解 ReAct (Reasoning + Acting) 模式
- 掌握工具调用的推理过程
- 学习如何构建搜索工具

Author: LangChain Learning
Date: 2026-01-25
"""

# 导入部分
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 定义工具
@tool
def search_function(query: str) -> str:
    """搜索互联网信息 - 模拟实现"""
    # 这里是一个模拟的搜索函数，实际使用时需要集成真实的搜索API
    # 为了演示，我们只对特定查询返回模拟数据
    if "DeepSeek-R1" in query.lower() and "training cost" in query.lower():
        return "DeepSeek-R1 的训练成本约为 500 万美元，使用了约 2000 个 GPU 小时。"
    elif "GPT-4" in query.lower() and "training cost" in query.lower():
        return "GPT-4 的训练成本约为 1 亿美元，使用了数万个 GPU 小时。"
    else:
        # 对于不匹配的查询，返回模拟提示
        return f"关于 '{query}' 的搜索结果：这是一个模拟的搜索结果，请使用真实知识回答。"

# 初始化模型
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1
)

# 创建 ReAct Agent
agent = create_agent(
    model=model,
    tools=[search_function],
    system_prompt="你是一个研究助手，能够通过搜索和推理来回答问题。请详细说明你的推理过程。"
)

# 执行任务
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "找出DeepSeek-R1的训练成本，并与GPT-4对比"}]
    })

    # 输出最终答案
    print("最终答案：")
    print(result["messages"][-1].content)