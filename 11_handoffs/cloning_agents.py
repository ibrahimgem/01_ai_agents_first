from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    tracing_disabled=True
)

hero_agent = Agent(
    name="Hero Agent",
    instructions="You are a hero agent."
)

robot_agent = hero_agent.clone(
    name="Robot Agent",
    instructions="You are a robot agent."
)

response = Runner.run_sync(starting_agent=robot_agent, input="Write like a robot", run_config=config)
print(response.last_agent)
print(response.final_output)