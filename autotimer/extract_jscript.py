"""PDF/DOCX script extraction module using Gemini AI."""

import os
from google import genai
from google.genai import types
from .utils import load_prompt, calculate_cost


def extract_jscript(pdf_path, api_key):
    """
    Extracts dialogue text from a Japanese PDF/DOCX script using Gemini.

    Args:
        pdf_path: Path to the input PDF/DOCX file.
        api_key: Gemini API Key.

    Returns:
        Extracted text string with ACTOR:TEXT lines.
    """
    client = genai.Client(api_key=api_key)

    print(f"  Reading script file: {os.path.basename(pdf_path)}...")
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    # Determine MIME type
    ext = os.path.splitext(pdf_path)[1].lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }
    mime_type = mime_map.get(ext, "application/pdf")

    # Load prompt from file
    prompt = load_prompt("extract_script.md")

    print("  Sending script to Gemini gemini-2.5-pro...")
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(
                data=pdf_data,
                mime_type=mime_type,
            ),
            prompt,
        ],
    )

    extracted_text = response.text or ""

    if response.usage_metadata:
        prompt_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        cost = calculate_cost("gemini-2.5-pro", prompt_tokens, output_tokens)
        print(f"  Tokens — Prompt: {prompt_tokens}, Output: {output_tokens}, "
              f"Total: {response.usage_metadata.total_token_count}")
        if cost is not None:
            print(f"  Cost   — ${cost:.4f}")

    line_count = len([l for l in extracted_text.strip().split("\n") if l.strip()])
    print(f"  Extraction complete: {line_count} dialogue lines found.")
    return extracted_text
