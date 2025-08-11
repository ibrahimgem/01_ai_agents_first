from typing import Any
from agents import Agent, Runner, RunConfig, RunContextWrapper, OpenAIChatCompletionsModel, AgentHooks, Tool, function_tool
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

class RunAgHooks(AgentHooks):
    def __init__(self, ag_display_name):
        self.ag_display_name = ag_display_name
        self.event_counter = 0
        
    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### Hello {self.ag_display_name}, {self.event_counter}, Agent: {agent} has been started. Usage: {context.usage}")
        
    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None :
        self.event_counter += 1
        print(f"### Hello {self.ag_display_name}, {self.event_counter}, Agent: {agent} has been ended. Usage: {context.usage}, Output: {output}")
    
    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(f"### {self.ag_display_name} {self.event_counter} Agent {source.name} handed off to {agent.name} ") 
    
    async def on_tool_start(self, context:RunContextWrapper, agent: Agent, tool: Tool) -> None:
        print(f"### {self.ag_display_name} {self.event_counter} Agent {agent.name} started tool {tool.name} ") 
        
    async def on_tool_end(self, context:RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        print(f"### {self.ag_display_name} {self.event_counter} Agent {agent.name} ended tool {tool.name} with result {result}") 
    
@function_tool

def multiply(num1: int, num2: int) -> int:
    return num1 * num2    

multiplier_agent = Agent(
    name = "Multiply Agent",
    instructions= "You will multiply the values and answer them.",
    hooks=RunAgHooks(ag_display_name="Multiplier Agent"),
    tools=[multiply],
)
    
    
triage_agent = Agent(
    name = "Triage Agent",
    instructions = "You are a triage agent. You can handoff and use tools.",
    hooks=RunAgHooks(ag_display_name="Triage Agent"),
    handoffs=[multiplier_agent]
)

async def main():
    response = await Runner.run(
        starting_agent=triage_agent,
        input="I want to multiply 6 by 8",
        run_config=config
    )
    print(response.final_output)
    
if __name__ == "__main__":
    asyncio.run(main())