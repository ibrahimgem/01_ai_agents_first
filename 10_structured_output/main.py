from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from pydantic import BaseModel
import os, asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

class CalenderEvent(BaseModel):
    name: str
    date: str  
    partipants: list[str]

async def main():
        
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    set_tracing_disabled(disabled=True)

    agent = Agent(
        name="Calender extractor",
        instructions="Extract the calendar event from the user input.",
        output_type=CalenderEvent,
        model=OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=client,
            ),
    )

    response = await Runner.run(
        starting_agent = agent,
        input= "There is an Independence Day celebration on 14th August 2025. There will be Ibrahim and Yasir in event.",
    )
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())