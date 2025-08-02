from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") 

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    tracing_disabled=True
)

async def main():
    agent=Agent(
        name="Joker",
        instructions="You are a helpful assistant."
    )
    
    response = Runner.run_streamed(agent, "Please crack 5 jokes.", run_config=config)
    async for event in response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end=" ", flush=True)            
if __name__ == "__main__":
    asyncio.run(main())