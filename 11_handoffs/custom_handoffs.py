from agents import (Agent, Runner, OpenAIChatCompletionsModel, RunConfig, handoff, RunContextWrapper)
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
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

# Customizing handoffs via the handoff() function

# def on_handoff(ctx: RunContextWrapper[None]):
#     print("Handoff called")
    
# agent = Agent(
#     name="My agent",
#     instructions="You are a helpful assistant who always says: 'Response from handoff agent.'")

# handoff_obj = handoff(
#     agent = agent,
#     on_handoff=on_handoff,
#     tool_name_override="custom_handoff_tool",
#     tool_description_override="This is custom handoff tool"
#     )


# general_agent = Agent(
#     name="General Agent",
#     instructions="Forward every query to the handoff agent.",
#     handoffs=[handoff_obj]
# )

# response = Runner.run_sync(general_agent, "Hello", run_config=config)
# print(response.final_output)

# Handoff inputs

class EscalationData(BaseModel):
    reason: str
    
def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent is called with reason: {input_data.reason}")
    
agent = Agent(name="Escalation Agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)

general_agent = Agent(
    name="General Agent", 
    instructions="Look into user's matters",
    handoffs=[handoff_obj],)

response = Runner.run_sync(general_agent, "I want escalation of my order.", run_config=config)