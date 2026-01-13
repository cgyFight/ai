import os
import tools
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
model = OpenAIChatModel(
    'qwen-plus',
    provider=OpenAIProvider(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key
    ),
    profile=OpenAIModelProfile(
        # Supported by OpenAIModel only, requires OpenAIModelProfile
        openai_supports_strict_tool_definition=False
    )
)
agent = Agent(model,
              system_prompt="You are an experienced programmer",
              tools=[tools.read_file, tools.list_files, tools.rename_file, tools.write_file])


def main():
    history = []
    while True:
        user_input = input("Input: ")
        resp = agent.run_sync(user_input,
                              message_history=history)
        history = list(resp.all_messages())
        print(resp.output)


if __name__ == "__main__":
    main()
