"""Manual demo: real model + real news DB, without the HTTP layer.

uv run --env-file .env python -m scripts.demo_news_loop "Tin mới về VCB?" --tier cheap
(Calls a paid provider; use the cheap tier.)
"""

import argparse
import asyncio

from app.core.settings import get_settings
from app.graph.workflow import run_agent
from app.integrations.llm.factory import get_chat_model
from app.tools.calculator.calculate import calculate_tool
from app.tools.sql.news_tool import search_news_tool


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--profile", default="deepseek")
    parser.add_argument("--tier", default="cheap")
    args = parser.parse_args()

    model = get_chat_model(args.profile, args.tier)  # type: ignore[arg-type]
    result = await run_agent(
        model,
        [search_news_tool, calculate_tool],
        args.question,
        max_steps=get_settings().agent_max_steps,
        max_retries=get_settings().verifier_max_retries,
        max_tool_calls=get_settings().agent_max_tool_calls,
    )

    for call in result.tool_calls:
        print(f"[tool] {call.name} {call.args} ok={call.ok}")
    print(result.answer)
    print(f"[verified] {result.verified} unsupported={result.unsupported}")


if __name__ == "__main__":
    asyncio.run(main())
