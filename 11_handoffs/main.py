from agents import (Agent, Runner, OpenAIChatCompletionsModel, RunConfig, handoff)
from openai import AsyncOpenAI
import rich

gemini_api_key = "AIzaSyDaoGdaRTiAqvoq1TxbnZTwKCz6SlPwnYw" 

external_model = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_model
)

config = RunConfig(
    model=model,
    model_provider=external_model,
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
    input="I need help with a python script.",
    run_config=config
)


print("Agent Responded:", result.last_agent)
rich.print("Final Result:", result.final_output)