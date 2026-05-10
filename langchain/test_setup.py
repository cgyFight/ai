#!/usr/bin/env python3
"""
01. LangChain 基础测试 - 验证环境配置

学习目标：
- 验证环境配置是否正确
- 测试基本的导入功能

Author: LangChain Learning
Date: 2025-01-24
"""

import os
from dotenv import load_dotenv


def test_imports():
    """测试基本导入功能"""
    try:
        print("📦 测试导入...")

        # 测试 LangChain 核心模块
        from langchain.agents import create_agent, AgentExecutor
        from langchain_anthropic import ChatAnthropic
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage

        print("✅ 所有导入成功！")
        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_environment():
    """测试环境变量"""
    try:
        print("🔍 检查环境变量...")
        load_dotenv()

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            print("✅ ANTHROPIC_API_KEY 已配置")
            return True
        else:
            print("❌ ANTHROPIC_API_KEY 未配置或使用默认值")
            print("📝 请在 .env 文件中设置正确的 API 密钥")
            return False

    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始环境验证...\n")

    # 测试导入
    imports_ok = test_imports()
    print()

    # 测试环境
    env_ok = test_environment()
    print()

    # 总结
    if imports_ok and env_ok:
        print("🎉 环境配置验证成功！可以运行 01_basic.py 了")
    else:
        print("⚠️  环境配置需要调整")
        print("\n📋 解决步骤：")
        print("1. 确保依赖已安装：uv sync")
        print("2. 配置 .env 文件中的 ANTHROPIC_API_KEY")
        print("3. 重新运行此测试脚本")


if __name__ == "__main__":
    main()
