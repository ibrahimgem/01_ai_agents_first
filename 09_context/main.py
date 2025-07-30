from agents import Agent, Runner, OpenAIChatCompletionsModel, RunContextWrapper, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from dataclasses import dataclass
import os, asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

@dataclass
class Context:
    city: str
    
@function_tool
async def get_weather(wrapper: RunContextWrapper[Context]) -> str:
        return f"{wrapper.context.city} me barish ho rahi hai."

async def main():
    
    my_context = Context(city="Karachi")
        
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    set_tracing_disabled(disabled=True)

    agent = Agent[Context](
        name="Weather Agent",
        instructions="You will be provided with a city name. Use the get_weather_user_city tool to provide the weather information.",
        tools=[get_weather],
        model=OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=client,
            ),
    )

    response = await Runner.run(
        starting_agent = agent,
        input= "What is the weather of user city?",
        context= my_context
    )
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())