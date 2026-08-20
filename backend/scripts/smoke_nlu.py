"""Manual live Gemini function-calling smoke test for M3."""

from __future__ import annotations

import argparse
import json

from app.core.config import settings
from app.services.nlu_service import AUTH_REQUIREMENT, parse_intent


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Gemini NLU smoke test")
    parser.add_argument("command", help="Vietnamese IT Helpdesk command")
    args = parser.parse_args()

    result = parse_intent(args.command)
    function_name = result["function_name"]
    print(f"Model: {settings.gemini_model}")
    print(f"Function: {function_name}")
    print(
        "Arguments: "
        + json.dumps(result["arguments"], ensure_ascii=False, sort_keys=True)
    )
    print(f"Auth requirement: {AUTH_REQUIREMENT[function_name]}")


if __name__ == "__main__":
    main()
