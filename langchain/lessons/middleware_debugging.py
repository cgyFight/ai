#!/usr/bin/env python3
"""
06. 使用 Middleware 进行 Logging 和 Debugging

学习目标：
- 理解 LangChain 中间件的工作原理
- 掌握如何使用中间件记录详细的执行信息
- 学习自定义中间件来监控工具调用、参数传递和返回结果

Author: LangChain Learning
Date: 2026-01-25
"""

# 导入部分
import os
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallWrapper,
)

# 加载环境变量
load_dotenv()

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    print(f"🔧 工具 get_weather 被调用，参数: city='{city}'")

    weather_data = {
        "北京": "晴天，温度 5°C",
        "上海": "多云，温度 8°C",
        "广州": "阴天，温度 15°C",
        "深圳": "小雨，温度 12°C"
    }

    result = weather_data.get(city, f"{city} 天气信息暂未收录")
    print(f"🔧 工具 get_weather 返回: '{result}'")
    return result

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    print(f"🔧 工具 calculate 被调用，参数: expression='{expression}'")

    try:
        result = eval(expression)
        final_result = f"{expression} = {result}"
        print(f"🔧 工具 calculate 返回: '{final_result}'")
        return final_result
    except Exception as e:
        error_msg = f"计算错误: {e}"
        print(f"🔧 工具 calculate 返回错误: '{error_msg}'")
        return error_msg

# 自定义 Logging 中间件
class DetailedLoggingMiddleware(AgentMiddleware[Dict[str, Any], Dict[str, Any]]):
    """详细的日志记录中间件"""

    def __init__(self, log_level: str = "INFO"):
        super().__init__()
        self.log_level = log_level
        self.execution_logs = []
        self.start_time = None

    def _log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """统一的日志记录方法"""
        if level == "DEBUG" and self.log_level != "DEBUG":
            return

        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data or {}
        }

        self.execution_logs.append(log_entry)

        # 控制台输出
        emoji = {"INFO": "ℹ️", "DEBUG": "🔍", "WARN": "⚠️", "ERROR": "❌"}
        print(f"{emoji.get(level, '📝')} [{timestamp}] {message}")

        if data and level == "DEBUG":
            print(f"   📊 数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """模型调用前的处理"""
        self.start_time = time.time()

        messages = state.get("messages", [])
        message_count = len(messages)

        self._log("INFO", f"准备调用模型，消息数量: {message_count}")

        return None

    def after_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """模型调用后的处理"""
        duration = time.time() - (self.start_time or time.time())

        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content[:200] + "..." if len(last_message.content) > 200 else last_message.content
                self._log("INFO", f"模型返回消息，耗时: {duration:.2f}s", {"content": content})

        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> Any:
        """包装工具调用 - 这是记录工具调用细节的关键方法"""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        self._log("INFO", f"准备调用工具: {tool_name}", {
            "tool_name": tool_name,
            "arguments": tool_args,
            "tool_call_id": request.tool_call.get("id", "unknown")
        })

        tool_start_time = time.time()

        try:
            # 调用工具
            result = handler(request)

            tool_duration = time.time() - tool_start_time

            # 记录工具调用结果
            if isinstance(result, ToolMessage):
                self._log("INFO", f"工具 {tool_name} 调用成功，耗时: {tool_duration:.2f}s", {
                    "tool_name": tool_name,
                    "result": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "tool_call_id": result.tool_call_id
                })
            else:
                self._log("INFO", f"工具 {tool_name} 返回结果，耗时: {tool_duration:.2f}s")

            return result

        except Exception as e:
            tool_duration = time.time() - tool_start_time
            self._log("ERROR", f"工具 {tool_name} 调用失败，耗时: {tool_duration:.2f}s", {
                "error": str(e),
                "tool_name": tool_name,
                "arguments": tool_args
            })
            raise

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        total_time = time.time() - (self.start_time or time.time())

        info_logs = [log for log in self.execution_logs if log["level"] == "INFO"]
        error_logs = [log for log in self.execution_logs if log["level"] == "ERROR"]
        debug_logs = [log for log in self.execution_logs if log["level"] == "DEBUG"]

        return {
            "total_execution_time": total_time,
            "total_logs": len(self.execution_logs),
            "info_logs": len(info_logs),
            "error_logs": len(error_logs),
            "debug_logs": len(debug_logs),
            "logs": self.execution_logs[-10:]  # 最近10条日志
        }

# 性能监控中间件
class PerformanceMonitoringMiddleware(AgentMiddleware[Dict[str, Any], Dict[str, Any]]):
    """性能监控中间件"""

    def __init__(self):
        super().__init__()
        self.metrics = {
            "model_calls": 0,
            "tool_calls": 0,
            "total_model_time": 0.0,
            "total_tool_time": 0.0,
            "errors": 0
        }
        self._model_start_time = None

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """记录模型调用开始"""
        self.metrics["model_calls"] += 1
        # 在实例上存储开始时间
        self._model_start_time = time.time()
        print(f"⚡ 模型调用开始")
        return None

    def after_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """记录模型调用结束"""
        if self._model_start_time:
            duration = time.time() - self._model_start_time
            self.metrics["total_model_time"] += duration
            print(f"⚡ 模型调用耗时: {duration:.2f}秒")
            self._model_start_time = None  # 重置
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """监控工具调用性能"""
        self.metrics["tool_calls"] += 1
        tool_name = request.tool_call.get("name", "unknown")

        start_time = time.time()
        try:
            result = handler(request)
            duration = time.time() - start_time
            self.metrics["total_tool_time"] += duration
            print(f"⚡ 工具 {tool_name} 调用成功，耗时: {duration:.2f}秒")
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.metrics["total_tool_time"] += duration
            self.metrics["errors"] += 1
            print(f"⚡ 工具 {tool_name} 调用失败，耗时: {duration:.2f}秒，错误: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        avg_model_time = self.metrics["total_model_time"] / max(self.metrics["model_calls"], 1)
        avg_tool_time = self.metrics["total_tool_time"] / max(self.metrics["tool_calls"], 1)

        return {
            **self.metrics,
            "avg_model_time": avg_model_time,
            "avg_tool_time": avg_tool_time,
            "success_rate": 1 - (self.metrics["errors"] / max(self.metrics["tool_calls"], 1))
        }

# 创建带有中间件的代理
def create_debugging_agent():
    """创建带有调试中间件的代理"""

    # 初始化模型
    model = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.1
    )

    # 创建中间件实例
    logging_middleware = DetailedLoggingMiddleware(log_level="DEBUG")
    performance_middleware = PerformanceMonitoringMiddleware()

    # 创建代理
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate],
        middleware=[logging_middleware, performance_middleware],
        system_prompt="""你是一个智能助手。我会详细记录所有的执行过程，包括工具调用、参数传递和返回结果。

请在需要时调用适当的工具来完成任务。"""
    )

    return agent, logging_middleware, performance_middleware

# 演示中间件调试功能
def demonstrate_middleware_debugging():
    """演示中间件调试功能"""

    print("=== 中间件调试演示 ===\n")

    agent, logging_middleware, performance_middleware = create_debugging_agent()

    conversations = [
        "今天北京的天气怎么样？",
        "请帮我计算 15 + 27 等于多少？",
        "现在请同时查询上海天气并计算 100 - 25"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n{'='*50}")
        print(f"第 {i} 轮对话")
        print(f"{'='*50}")

        print(f"👤 用户: {user_input}\n")

        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=user_input)]
            })

            print(f"\n🤖 助手: {result['messages'][-1].content}")

        except Exception as e:
            print(f"\n❌ 执行出错: {e}")

    # 显示执行摘要
    print(f"\n{'='*50}")
    print("📊 执行摘要")
    print(f"{'='*50}")

    # 日志摘要
    summary = logging_middleware.get_execution_summary()
    print(f"总执行时间: {summary['total_execution_time']:.2f}秒")
    print(f"日志条数: {summary['total_logs']} (INFO: {summary['info_logs']}, ERROR: {summary['error_logs']}, DEBUG: {summary['debug_logs']})")

    # 性能指标
    metrics = performance_middleware.get_metrics()
    print(f"\n⚡ 性能指标:")
    print(f"  模型调用: {metrics['model_calls']} 次")
    print(f"  工具调用: {metrics['tool_calls']} 次")
    print(f"  平均模型耗时: {metrics['avg_model_time']:.2f}秒")
    print(f"  平均工具耗时: {metrics['avg_tool_time']:.2f}秒")
    print(f"  成功率: {metrics['success_rate']:.1%}")
    print(f"  错误次数: {metrics['errors']}")

    # 显示最近的日志
    print(f"\n📝 最近日志:")
    for log in summary['logs'][-5:]:  # 显示最后5条日志
        print(f"  {log['timestamp'][:19]} [{log['level']}] {log['message']}")

# 运行演示
if __name__ == "__main__":
    demonstrate_middleware_debugging()