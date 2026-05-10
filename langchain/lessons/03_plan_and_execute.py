#!/usr/bin/env python3
"""
03. Plan-and-Execute 代理 - 规划与执行模式

学习目标：
- 理解 Plan-and-Execute 模式的工作原理
- 掌握先规划后执行的代理设计
- 学习如何构建多步骤任务执行

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
def research_topic(topic: str) -> str:
    """研究特定主题，收集相关信息"""
    # 模拟研究工具
    research_data = {
        "AI发展": "人工智能正在快速发展，预计到2030年市场规模将达到5000亿美元",
        "机器学习": "机器学习是AI的核心技术，包括监督学习、无监督学习和强化学习",
        "深度学习": "深度学习使用神经网络模拟人脑处理信息，已在图像识别等领域取得突破",
        "自然语言处理": "NLP技术让机器理解和生成人类语言，应用包括翻译、问答系统等"
    }
    return research_data.get(topic, f"关于'{topic}'的研究结果：这是一个模拟的调研数据。")

@tool
def analyze_data(data: str) -> str:
    """分析收集到的数据，提取关键洞察"""
    # 模拟分析工具
    if "AI发展" in data:
        return "关键洞察：AI市场增长迅速，但需要关注伦理和就业影响"
    elif "机器学习" in data:
        return "关键洞察：监督学习在商业应用中最广泛，无监督学习潜力巨大"
    elif "深度学习" in data:
        return "关键洞察：深度学习在CV和NLP领域领先，但计算资源需求高"
    elif "自然语言处理" in data:
        return "关键洞察：NLP技术正在改变人机交互方式，多语言支持是未来趋势"
    else:
        return f"数据分析结果：{data[:100]}...（分析完成）"

@tool
def write_report(topic: str, insights: str) -> str:
    """基于洞察编写报告"""
    return f"""
# {topic} 研究报告

## 主要发现
{insights}

## 结论
基于以上分析，{topic} 领域正在快速发展，建议加大投入和关注。

## 建议
1. 持续跟踪技术发展
2. 评估潜在应用场景
3. 关注相关政策变化
"""

# 初始化模型
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
    timeout=30  # 添加超时设置
)

# 创建 Plan-and-Execute Agent
agent = create_agent(
    model=model,
    tools=[research_topic, analyze_data, write_report],
    system_prompt="""你是一个项目经理，使用 Plan-and-Execute 模式工作。

工作流程：
1. 首先制定详细的执行计划，列出所有需要完成的步骤
2. 按计划顺序执行每个步骤，不要跳跃
3. 每个步骤完成后，记录结果并继续下一步
4. 只有在所有步骤完成后才给出最终答案

请始终遵循：先规划，再执行的模式。"""
)

# 执行任务
if __name__ == "__main__":
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "请简单分析机器学习的发展趋势"}]
        })

        # 输出最终答案
        print("Plan-and-Execute 执行结果：")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"执行出错: {e}")
        print("可能是网络超时，请检查网络连接或 API 密钥")