from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Literal
from agents import Agent, ItemHelpers, Runner, OpenAIChatCompletionsModel, RunConfig, TResponseInputItem, trace
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv ()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client = client,
)

config = RunConfig(
    model = model,
)

blog_outline_generator = Agent(
    name="blog_outline_generator",
    instructions=(
        "You generate a blog post outline based on the user's input. "
        "If there is any feedback provided, use it to improve the outline."
    ),
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate a blog outline and decide if it's good enough. "
        "If it's not good enough, you provide feedback on what needs to be improved. "
        "Never give it a pass on the first try. After 5 attempts, you can give it a pass if the blog outline is good enough - do not go for perfection"
    ),
    output_type=EvaluationFeedback,
)


async def main() -> None:
    msg = input("What kind of blog would you like to write? ")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_outline: str | None = None

    # We'll run the entire workflow in a single trace
    with trace("LLM as a judge"):
        while True:
            blog_outline_result = await Runner.run(
                blog_outline_generator,
                input_items,
                run_config=config,
            )

            input_items = blog_outline_result.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(blog_outline_result.new_items)
            print("Blog post outline generated")

            evaluator_result = await Runner.run(evaluator, input_items, run_config=config)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")

            if result.score == "pass":
                print("Blog post outline is good enough, exiting.")
                break

            print("Re-running with feedback")

            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"Final blog post outline: {latest_outline}")


if __name__ == "__main__":
    asyncio.run(main())