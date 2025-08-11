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

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client,
)

config = RunConfig(
    model=model,
    tracing_disabled=True,
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

response = Runner.run_sync(agent, "Hello! Where is Karachi located?", run_config=config)
print(response.final_output)