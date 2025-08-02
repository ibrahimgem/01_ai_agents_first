from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
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

deposit_agent = Agent(
    name="Deposit Agent",
    instructions="You will help users in depositing queries."
)

withdraw_agent = Agent(
    name="Withdraw Agent",
    instructions="You will help users in withdrawal queries."
)

triage_agent = Agent(
    name= "Triage Agent",
    instructions= "You are handoff the task to specialist agents and help with users questions.",
    handoffs= [deposit_agent, withdraw_agent]
)

async def main():

    response= await Runner.run(
        starting_agent=triage_agent, 
        input= "I need help in depositing the payment.", 
        run_config=config
    )
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())