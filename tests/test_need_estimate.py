"""fit_context_word_estimate: per-word need estimate against measured token counts."""

from __future__ import annotations

import base64
import json
import random
import string

import pytest

from laplace_serve._token_words import COST1, COST2
from laplace_serve.app import _TEMPLATE_ALLOWANCE_TOKENS, _estimate_need_tokens
from laplace_serve.config import LaplacedConfig

_TOOL = {"type": "function", "function": {"name": "f", "description": "d", "parameters": {
    "type": "object", "properties": {"a": {"type": "string"}}}}}


def need_corpus() -> dict[str, tuple[list, list | None]]:
    """Seeded adversarial requests: name -> (messages, tools). Synthetic only.

    scripts/measure-need-estimate.py renders each one through five local chat
    templates and prints MEASURED below. Regenerating the word lists changes
    these bodies, so re-measure after scripts/gen-token-words.py.
    """
    rng = random.Random(1115)
    one, two = sorted(COST1), sorted(COST2)
    both = one + two

    def rs(alphabet, n):
        return "".join(rng.choice(alphabet) for _ in range(n))

    def user(text):
        return [{"role": "user", "content": text}]

    def rand_json(depth):
        if depth == 0 or rng.random() < 0.3:
            return rng.choice([rng.choice(both), rs(string.ascii_letters, 5), rng.randint(-9999, 9999), True, None])
        if rng.random() < 0.5:
            return {rng.choice(both): rand_json(depth - 1) for _ in range(rng.randint(1, 5))}
        return [rand_json(depth - 1) for _ in range(rng.randint(1, 5))]

    def code_line():
        a, b, c, d = (rng.choice(both) for _ in range(4))
        return rng.choice([
            f"{' ' * 4 * rng.randint(0, 4)}{a} = {b}.{c}({d}, {rng.randint(0, 99)})",
            f"{' ' * 4 * rng.randint(0, 4)}if {a} is not None and {b}[{c!r}]:",
            f"{' ' * 4 * rng.randint(0, 4)}def {a}_{b}(self, {c}={d}):",
            f"{' ' * 4 * rng.randint(0, 4)}# {a.capitalize()} {b} {c} {d}.",
        ])

    pieces = [lambda: rng.choice(one), lambda: rng.choice(two).capitalize(), lambda: rs(string.ascii_letters, 3),
              lambda: rs(string.digits, 2), lambda: rs(string.punctuation, 1), lambda: " ", lambda: "  ",
              lambda: "\n", lambda: "\t", lambda: "é", lambda: "中", lambda: "\U0001f642"]
    return {
        "alnum12k": (user(rs(string.ascii_letters + string.digits, 12000)), None),
        "digits12k": (user(rs(string.digits, 12000)), None),
        "punct12k": (user(rs(string.punctuation, 12000)), None),
        "cjk4k": (user(rs([chr(c) for c in range(0x4E00, 0x9FFF)], 4000)), None),
        "emoji3k": (user(rs([chr(c) for c in range(0x1F300, 0x1F5FF)], 3000)), None),
        "empty_msgs": ([{"role": ("user", "assistant")[i % 2], "content": ""} for i in range(400)], None),
        "word_msgs": ([{"role": ("user", "assistant")[i % 2], "content": rng.choice(both)} for i in range(400)], None),
        "text_parts": ([{"role": "user", "content": [{"type": "text", "text": " ".join(rng.choice(one) for _ in range(40))}
                                                     for _ in range(60)]}], None),
        "tools_min": (user("hi"), [_TOOL]),
        "tool_params_words": (user("hi"), [{"type": "function", "function": {
            "name": "g", "description": "", "parameters": {"type": "object", "properties": {
                w: {"type": "string", "description": w} for w in one[:200]}, "required": one[:200]}}}]),
        "base64": (user(base64.b64encode(bytes(rng.getrandbits(8) for _ in range(9000))).decode()), None),
        "base64_lines": (user("\n".join(base64.b64encode(bytes(rng.getrandbits(8) for _ in range(54))).decode()
                                        for _ in range(170))), None),
        "hex12k": (user(rs("0123456789abcdef", 12000)), None),
        "minified_json": (user(json.dumps(rand_json(7), separators=(",", ":"))), None),
        "code": (user("\n".join(code_line() for _ in range(600))), None),
        "random_lower_words": (user(" ".join(rs(string.ascii_lowercase, rng.randint(1, 12)) for _ in range(1500))), None),
        "random_cap_words": (user(" ".join(rs(string.ascii_lowercase, rng.randint(1, 12)).capitalize()
                                           for _ in range(1500))), None),
        "cost1_words": (user(" ".join(rng.choice(one) for _ in range(4000))), None),
        "cost1_words_cap": (user(" ".join(rng.choice(one).capitalize() for _ in range(4000))), None),
        "cost2_words": (user(" ".join(rng.choice(two) for _ in range(3000))), None),
        "cost2_words_cap": (user(" ".join(rng.choice(two).capitalize() for _ in range(3000))), None),
        "words_newlines": (user("\n".join(rng.choice(both) for _ in range(4000))), None),
        "words_punct": (user("".join(rng.choice(both) + rng.choice(string.punctuation) for _ in range(3000))), None),
        "words_quoted": (user(" ".join(f'"{rng.choice(both)}"' for _ in range(3000))), None),
        "words_mixed": (user("".join(rng.choice(pieces)() for _ in range(8000))), None),
        "single_letters": (user(" ".join(rs(string.ascii_lowercase, 1) for _ in range(6000))), None),
        "space_runs": (user("".join(" " * rng.randint(1, 70) + rng.choice(both) for _ in range(800))), None),
        "escapes": (user("".join(rng.choice(["\n", "\t", '"', "\\", "/", "\nthe", "\tself", '"the"', "éthe"])
                                 for _ in range(4000))), None),
        "tool_loop": ([{"role": "system", "content": " ".join(rng.choice(both) for _ in range(200))}] + [
            m for i in range(80) for m in (
                {"role": "assistant", "content": " ".join(rng.choice(both) for _ in range(10)), "tool_calls": [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": "f", "arguments": json.dumps({"a": rng.choice(both)})}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": "\n".join(code_line() for _ in range(15))},
            )], [_TOOL]),
    }


def corpus_body(messages, tools) -> bytes:
    payload = {"model": "m", "messages": messages, "max_tokens": 1}
    if tools:
        payload["tools"] = tools
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


# Highest prompt-token count over GLM-4.5-Air, gpt-oss-120b, Qwen3-Coder-30B,
# Qwen3-Next-80B and Gemma 4 26B (chat template applied, generation prompt on),
# from scripts/measure-need-estimate.py on 2026-11-15, and the tokenizer that set
# it (short names for the MLX model directories under ~/.lmstudio/models).
MEASURED = {
    "alnum12k": (8851, "qwen3-next"),
    "digits12k": (12013, "gemma4"),
    "punct12k": (8075, "gpt-oss"),
    "cjk4k": (8415, "gemma4"),
    "emoji3k": (8713, "glm"),
    "empty_msgs": (2063, "gpt-oss"),
    "word_msgs": (2589, "gpt-oss"),
    "text_parts": (2526, "gpt-oss"),
    "tools_min": (257, "qwen3-coder"),
    "tool_params_words": (6238, "qwen3-coder"),
    "base64": (8893, "qwen3-next"),
    "base64_lines": (9319, "qwen3-next"),
    "hex12k": (10613, "gemma4"),
    "minified_json": (1024, "qwen3-next"),
    "code": (7340, "gemma4"),
    "random_lower_words": (5543, "qwen3-next"),
    "random_cap_words": (5976, "qwen3-next"),
    "cost1_words": (4067, "gpt-oss"),
    "cost1_words_cap": (4067, "gpt-oss"),
    "cost2_words": (4203, "gpt-oss"),
    "cost2_words_cap": (5328, "gpt-oss"),
    "words_newlines": (9442, "gpt-oss"),
    "words_punct": (6755, "gpt-oss"),
    "words_quoted": (10096, "gpt-oss"),
    "words_mixed": (9104, "gemma4"),
    "single_letters": (6067, "gpt-oss"),
    "space_runs": (2283, "gemma4"),
    "escapes": (5395, "gemma4"),
    "tool_loop": (18592, "gpt-oss"),
}


def test_word_estimate_flag_is_opt_in_and_validated():
    assert LaplacedConfig().fit_context_word_estimate is False
    assert LaplacedConfig.from_dict({"fit_context_word_estimate": True}).fit_context_word_estimate is True
    with pytest.raises(ValueError, match="fit_context_word_estimate"):
        LaplacedConfig.from_dict({"fit_context_word_estimate": "yes"})


def test_word_estimate_off_is_byte_estimate():
    body = corpus_body([{"role": "user", "content": "the value of self"}], None)
    # The body's max_tokens (1) is the output allowance.
    assert _estimate_need_tokens(body, 7) == len(body) + _TEMPLATE_ALLOWANCE_TOKENS + 1
    assert _estimate_need_tokens(body, 7, False) == _estimate_need_tokens(body, 7)


def test_word_estimate_covers_measured_corpus():
    corpus = need_corpus()
    assert sorted(corpus) == sorted(MEASURED)
    for name, (messages, tools) in corpus.items():
        body = corpus_body(messages, tools)
        real, tokenizer = MEASURED[name]
        estimate = _estimate_need_tokens(body, 1, True) - 1
        assert estimate >= real, (name, tokenizer, estimate, real)
        assert estimate <= _estimate_need_tokens(body, 1) - 1


def test_word_estimate_discounts_only_message_text():
    # Tool schemas and tool-call arguments stay at bytes: templates can expand
    # them (Qwen3-Coder renders each parameter as XML).
    tools = [{"type": "function", "function": {"name": "read", "description": "the file path",
                                               "parameters": {"type": "object"}}}]
    call = [{"id": "c", "type": "function",
             "function": {"name": "read", "arguments": json.dumps({"path": "the file"})}}]
    body = corpus_body([{"role": "assistant", "content": "", "tool_calls": call}], tools)
    assert _estimate_need_tokens(body, 1, True) == _estimate_need_tokens(body, 1)
    # Non-JSON and non-chat bodies keep the byte estimate.
    assert _estimate_need_tokens(b"the the the", 1, True) == _estimate_need_tokens(b"the the the", 1)
    legacy = json.dumps({"prompt": "the value of self"}).encode()
    assert _estimate_need_tokens(legacy, 1, True) == _estimate_need_tokens(legacy, 1)


def test_word_estimate_word_rules():
    def prompt(text):
        return _estimate_need_tokens(corpus_body([{"role": "user", "content": text}], None), 1, True)

    base = prompt("")
    assert "the" in COST1 and "value" in COST1
    assert prompt("the value") - base == 2  # " value": the space is free
    assert prompt("The Value") - base == 2  # capitalised words count too
    assert prompt("THE") - base == 3  # other casing keeps bytes
    assert prompt("the1") - base == 4  # glued to a digit: bytes
    assert prompt("thé") - base == 4  # glued to non-ASCII: bytes
    assert prompt(" " * 33) - base == 3  # ceil(33 / 16)
    two = sorted(w for w in COST2 if len(w) >= 6)[0]
    assert prompt(two) - base == 2
    # JSON escapes are not discounted: "\n" is two body bytes and stays two.
    assert prompt("the\nvalue") - base == 4


def test_word_estimate_tightens_english():
    text = (
        "The broker sizes each model load from an estimate of the request's "
        "context need. When the estimate is too high, a large job loads at the "
        "configured ceiling, which uses more memory than the job needs and can "
        "force other models out. This test only checks that ordinary English "
        "text is estimated well below one token per byte. "
    ) * 40
    body = corpus_body([{"role": "user", "content": text}], None)
    words = _estimate_need_tokens(body, 1, True) - _TEMPLATE_ALLOWANCE_TOKENS
    assert words < 0.5 * (_estimate_need_tokens(body, 1) - _TEMPLATE_ALLOWANCE_TOKENS)
