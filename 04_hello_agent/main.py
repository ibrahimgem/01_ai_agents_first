# #AGENT LEVEL

# import asyncio
# from openai import AsyncOpenAI
# from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
# import os
# gemini_api_key = os.getenv("gemini_api_key")


# client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# set_tracing_disabled(disabled=True)

# async def main():
#     agent = Agent(
#         name="Assistant",
#         instructions="You only respond in haikus.",
#         model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
#     )
    
#     result = await Runner.run(
#         agent,
#         "What is AI Agents?",
#     )
#     print(result.final_output)
    
# if __name__ == "__main__":
#     asyncio.run(main())
    
    
    
# RUN LEVEL

# import asyncio
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
# from agents.run import RunConfig
# from dotenv import load_dotenv
# import os

# load_dotenv()

# gemini_api_key = os.getenv("gemini_api_key")


# external_client = AsyncOpenAI(
#     api_key = gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash",
#     openai_client=external_client
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# result = Runner.run_sync(agent, "Hello, how are you.", run_config=config)

# print(result.final_output)


# GLOBAL LEVEL

from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("gemini_api_key")

set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="gemini-2.0-flash")

result = Runner.run_sync(agent, "Hello")

print(result.final_output)