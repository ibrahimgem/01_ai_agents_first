from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, RunHooks, RunContextWrapper
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client=client,
)

config = RunConfig(
    model=model,
    tracing_disabled=True,
)

class TestHooks(RunHooks):
    def __init__(self):
        self.name = "TestHooks"
        self.counter = 0
        
    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.counter += 1
        print(f"This is {self.name} and the counter is {self.counter} from {agent.name} and {context.usage}")

start_hook = TestHooks()
        
agent = Agent(
    name="Assistant",
    instructions="You are a helpful agent.",
)

async def main():

    response = await Runner.run(
        agent,
        "What is whatsapp?",
        run_config=config,
        hooks=start_hook,
    )
    
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())