from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, ModelSettings, function_tool
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client=client,
)

config = RunConfig(
    model=model
)

# Temperature - The Creativity Knob


# Low temperature (0.1) = Very focused, consistent answers
agent_focused = Agent(
    name="Math Tutor",
    instructions="You are a precise math tutor.",
    model_settings=ModelSettings(temperature=0.1)
)

# High temperature (0.9) = More creative, varied responses
agent_creative = Agent(
    name="Story Writer",
    instructions="You are a creative storyteller.",
    model_settings=ModelSettings(temperature=0.9)
)

# Tool Choice - The "Can I Use Tools?" Switch

@function_tool
def calculator(num1: int, num2: int) -> int:
    return num1, num2

@function_tool
def weather_tool(location: str) -> str:
    return f"The weather in {location} is rainy with 29°C"

# Agent can decide when to use tools (default)
agent_auto = Agent(
    name="Smart Assistant",
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(tool_choice="auto")
)

# Agent MUST use a tool (even if not needed)
agent_required = Agent(
    name="Tool User",
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(tool_choice="required")
)

# Agent CANNOT use tools (chat only)
agent_no_tools = Agent(
    name="Chat Only",
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(tool_choice="none")
)

# Max Tokens - The Response Length Limit

# Short, concise responses
agent_brief = Agent(
    name="Brief Assistant",
    model_settings=ModelSettings(max_tokens=100)
)

# Longer, detailed responses
agent_detailed = Agent(
    name="Detailed Assistant", 
    model_settings=ModelSettings(max_tokens=1000)
)

msg = input(f"Hello I am {agent_detailed}, How may I help you? ")

response = Runner.run_sync(agent_detailed, msg, run_config=config)
print(response.final_output)