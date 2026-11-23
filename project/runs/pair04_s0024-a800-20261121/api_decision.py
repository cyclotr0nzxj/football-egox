"""Request one bounded execution decision using an explicitly sanitized prompt."""

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_stream(response):
    parts = []
    for line in response:
        line = line.rstrip(b"\r\n")
        if line.startswith(b"data:"):
            parts.append(line[5:].lstrip())
        elif not line and parts:
            event_data = b"\n".join(parts)
            parts.clear()
            if event_data == b"[DONE]":
                break
            event = json.loads(event_data)
            if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                return event["response"]
            if event.get("type") == "error":
                raise ValueError("Responses stream reported an error.")
    raise ValueError("Responses stream ended without a final response.")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token_cap", nargs="?", type=int, default=12000)
    parser.add_argument("--model", choices=("gpt-6.2-sol", "gpt-6.2-astra"), default="gpt-6.2-sol")
    parser.add_argument("--timeout", type=int, choices=(100, 160, 300), default=100)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    if not 1500 <= args.token_cap <= 32000:
        print(json.dumps({"status": "invalid_output_cap"}))
        return 2
    prompt = sys.stdin.read(6001).strip()
    if not prompt or len(prompt) > 6000:
        print(json.dumps({"status": "invalid_prompt_length"}))
        return 2
    try:
        env_path = Path(__file__).resolve().parents[2] / ".env.local"
        key, = (
            line.partition("=")[2].strip().strip("'\"")
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("OPENAI_API_KEY=")
        )
        if not key:
            raise ValueError
    except (OSError, ValueError):
        print(json.dumps({"status": "existing_config_unavailable"}))
        return 2

    body = json.dumps({
        "model": args.model,
        "reasoning": {"effort": "max"},
        "service_tier": "default",
        "store": False,
        "stream": args.stream,
        "max_output_tokens": args.token_cap,
        "input": prompt,
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=args.timeout) as response:
            result = read_stream(response) if args.stream else json.load(response)
    except urllib.error.HTTPError as error:
        print(json.dumps({"status": f"http_{error.code}"}))
        return 2
    except (OSError, ValueError) as error:
        cause = getattr(error, "reason", error)
        print(json.dumps({"status": "transport_error", "error_type": type(cause).__name__}))
        return 2

    model = result.get("model", "")
    status = result.get("status")
    usage = result.get("usage") or {}
    print(json.dumps({
        "model": model if re.fullmatch(r"gpt-[a-z0-9.-]{1,70}", model) else None,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        **({"reasoning_tokens": usage["output_tokens_details"]["reasoning_tokens"]}
           if isinstance((usage.get("output_tokens_details") or {}).get("reasoning_tokens"), int) else {}),
        "status": status if status in {"completed", "incomplete", "failed"} else "unexpected",
    }))
    if status != "completed" or result.get("service_tier") != "default":
        return 2
    print("\n".join(
        content["text"] for message in result.get("output", []) if message.get("type") == "message"
        for content in message.get("content", []) if content.get("type") == "output_text"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
