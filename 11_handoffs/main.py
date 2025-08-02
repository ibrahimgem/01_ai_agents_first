from agents import (Agent, Runner, OpenAIChatCompletionsModel, RunConfig)
from openai import AsyncOpenAI
import rich
from dotenv import load_dotenv
import os
load_dotenv()

# Load the Gemini API key from environment variables
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

next_js_assistant = Agent(
    name="Next.js Assistant",
    instructions="An AI assistant that helps with Next.js development tasks.",
)

python_assistant = Agent(
    name="Python Assistant",
    instructions="An AI assistant that helps with Python development tasks.",
)

triage_assistant = Agent(
    name="Triage Assistant",
    instructions="An AI assistant that triages tasks and decides which assistant to hand off to.",
    handoffs=[next_js_assistant, python_assistant]
)

result = Runner.run_sync(
    starting_agent=triage_assistant,  
    input="How can I create a new project in Next.js?",
    run_config=config
)


print("Agent Responded:", result.last_agent)
rich.print("Final Result:", result.final_output)