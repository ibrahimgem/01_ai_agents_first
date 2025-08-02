from agents import (Agent, Runner, OpenAIChatCompletionsModel, RunConfig, handoff)
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client= client
)

config = RunConfig(
    model = model,
    tracing_disabled = True
)

def billing_order():
    return(f"You billing has been processed.")

billing_agent = Agent(name="Billing Agent")

general_agent = Agent(
    name = "General Agent",
    instructions = "You are a General Agent. You will handoff to specilist agents.",
    handoffs = [
        handoff(
        agent=billing_agent,
        tool_name_override="billing_order",
        tool_description_override="Assists in billing matters."
        )]
    )

result = Runner.run_sync(
    starting_agent=general_agent, 
    input="I need help with billing.",
    run_config=config)
print(result.final_output)
print(f"Hello this is {result.last_agent.name}.")