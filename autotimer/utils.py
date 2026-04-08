"""Utility functions for AutoTimer."""

import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


# Pricing per 1M tokens (USD), as of April 2026
# Source: https://ai.google.dev/gemini-api/docs/pricing
_PRICING = {
    "gemini-2.5-pro": {
        "input_low":   1.25,   # prompt <= 200k tokens
        "input_high":  2.50,   # prompt >  200k tokens
        "output_low":  10.00,  # prompt <= 200k tokens (includes thinking)
        "output_high": 15.00,  # prompt >  200k tokens
        "threshold":   200_000,
    },
    "gemini-3-flash-preview": {
        "input":  0.50,  # text / image / video
        "output": 3.00,  # includes thinking tokens
    },
}


def calculate_cost(model, prompt_tokens, output_tokens):
    """
    Calculates the estimated API cost for a Gemini call.

    Args:
        model: Model name string (e.g. 'gemini-2.5-pro').
        prompt_tokens: Number of input tokens from usage_metadata.
        output_tokens: Number of output/candidate tokens from usage_metadata.

    Returns:
        Cost in USD as a float, or None if the model is not in the pricing table.
    """
    pricing = _PRICING.get(model)
    if not pricing:
        return None

    if "threshold" in pricing:
        if prompt_tokens <= pricing["threshold"]:
            input_cost = prompt_tokens * pricing["input_low"] / 1_000_000
            output_cost = output_tokens * pricing["output_low"] / 1_000_000
        else:
            input_cost = prompt_tokens * pricing["input_high"] / 1_000_000
            output_cost = output_tokens * pricing["output_high"] / 1_000_000
    else:
        input_cost = prompt_tokens * pricing["input"] / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000

    return input_cost + output_cost


def load_prompt(prompt_name):
    """
    Loads a prompt from the autotimer/prompts directory.
    Uses pkg_resources/importlib.resources if possible, with a filesystem fallback.

    Args:
        prompt_name: Filename of the prompt (e.g., 'extract_script.md').

    Returns:
        The content of the prompt as a string.
    """
    try:
        from . import prompts
        return pkg_resources.read_text(prompts, prompt_name)
    except Exception:
        # Fallback if package structure is not yet fully recognized/installed
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_name)
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
