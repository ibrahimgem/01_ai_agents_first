from typing import Any, List
from agents import (Agent, FunctionToolResult, RunContextWrapper, Runner, RunConfig, OpenAIChatCompletionsModel, function_tool, ModelSettings)
from openai import AsyncOpenAI
from agents.agent import StopAtTools, ToolsToFinalOutputResult
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
    openai_client=client
)

config = RunConfig(
    model=model,
    tracing_disabled=True
)

@function_tool
async def get_weather(city: str) -> str:
    return(f"The weather in {city} is sunny with 35°C")

@function_tool
async def calc(num1: int, num2: int) -> int:
    return(f"The answer of {num1} + {num2} is {num1+num2}")

def custom_tool_handler(
    context: RunContextWrapper[Any],
    tool_results: List[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """Processes tool results to decide final output."""
    for result in tool_results:
        if result.output and "sunny" in result.output:
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output=f"Final weather: {result.output}"
            )
    return ToolsToFinalOutputResult(
        is_final_output=False,
        final_output=None
    )


agent = Agent(
    name="Assistant",
    instructions="You are helpul assistant.",
    tools=[get_weather, calc],
    # model_settings=ModelSettings(tool_choice="none")
    # tool_use_behavior="stop_on_first_tool"
    # tool_use_behavior=StopAtTools(stop_at_tool_names=["calc"])
    tool_use_behavior=custom_tool_handler
)

response = Runner.run_sync(
    starting_agent=agent,
    input="What is the weather in Karachi? Also sum 7 and 6.",
    run_config=config
)

print(response.final_output)