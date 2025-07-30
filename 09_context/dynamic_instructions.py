from agents import Agent, Runner, OpenAIChatCompletionsModel, RunContextWrapper, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from dataclasses import dataclass
import os, asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

@dataclass
class Context:
    name: str
    city: str 

def dynamic_instructions(wrapper: RunContextWrapper[Context], agent: Agent[Context]) -> str:
        return f"The user's name is {wrapper.context.name} he is from {wrapper.context.city} and help him with his questions."

async def main():
    
    my_context = Context(name="Muhammad Ibrahim", city="Karachi")
        
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    set_tracing_disabled(disabled=True)

    agent = Agent[Context](
        name="Assistant",
        instructions=dynamic_instructions,
        model=OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=client,
            ),
    )

    response = await Runner.run(
        starting_agent = agent,
        input= "What is name of my city.?",
        context= my_context
    )
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())