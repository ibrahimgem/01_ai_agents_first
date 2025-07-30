from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Client for Gemini API

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_tracing_disabled(disabled=True)

# Define a function tool to fetch weather information
@function_tool
async def fetch_weather(location: str) -> str:
    return f"The weather in {location} is sunny with 40°C."

async def main():
    # This agent will use custom model and tools
    agent = Agent(
        name="WeatherAgent",
        instructions="You are a helpful assistant that provides weather information.",
        model=OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=client,
        ),
        tools=[fetch_weather],
    )
    
    response = await Runner.run(agent, "Hello, can you tell me the weather in Karachi?")
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())