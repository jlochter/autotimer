"""Utility functions for AutoTimer."""

import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


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
