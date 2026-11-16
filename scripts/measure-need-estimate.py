#!/usr/bin/env python3
"""Measure real prompt tokens for tests/test_need_estimate.py's corpus. Offline.

Needs ``transformers`` and ``jinja2`` (no torch) and local tokenizer files; it
never calls LM Studio. For each corpus request it applies each model's chat
template (generation prompt on) and prints a MEASURED entry: the highest count
and the tokenizer that set it. It also prints the per-tokenizer count, the byte
estimate and the word estimate, so a reviewer can see every margin.

Templates that reject the request as sent are retried with string tool-call
arguments parsed to a dict (GLM-4.5-Air and Qwen3-Coder need this) and
list-of-parts content joined with newlines into one string.

Usage: measure-need-estimate.py MODEL_DIR...   (run from the repo root)
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from laplace_serve.app import _estimate_need_tokens  # noqa: E402
from tests.test_need_estimate import corpus_body, need_corpus  # noqa: E402


def render(tok, messages, tools):
    kwargs = {"tools": tools} if tools else {}

    def count(msgs):
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kwargs)
        ids = getattr(ids, "input_ids", ids)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return len(ids)

    try:
        return count(messages)
    except Exception:
        parsed = copy.deepcopy(messages)
        for message in parsed:
            for call in message.get("tool_calls") or []:
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            if isinstance(message.get("content"), list):
                message["content"] = "\n".join(part["text"] for part in message["content"])
        return count(parsed)


def main(dirs):
    if not dirs:
        sys.exit(__doc__)
    from transformers import AutoTokenizer

    tokenizers = {
        os.path.basename(d.rstrip("/")): AutoTokenizer.from_pretrained(d, local_files_only=True)
        for d in dirs
    }
    measured = {}
    for name, (messages, tools) in need_corpus().items():
        body = corpus_body(messages, tools)
        counts = {t: render(tok, messages, tools) for t, tok in tokenizers.items()}
        top = max(counts, key=counts.get)
        measured[name] = (counts[top], top)
        words = _estimate_need_tokens(body, 1, True) - 1
        print(f"# {name}: bytes_est={_estimate_need_tokens(body, 1) - 1} word_est={words} "
              f"margin={words - counts[top]} counts={counts}", file=sys.stderr)
    print("MEASURED = {")
    for name, (count, top) in measured.items():
        print(f'    "{name}": ({count}, "{top}"),')
    print("}")


if __name__ == "__main__":
    main(sys.argv[1:])
