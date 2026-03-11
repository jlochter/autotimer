"""PDF/DOCX script extraction module using Gemini AI."""

import os
from google import genai
from google.genai import types
from .utils import load_prompt


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

    print("  Sending script to Gemini gemini-3-flash-preview...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
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
        print(f"  Tokens — Prompt: {response.usage_metadata.prompt_token_count}, "
              f"Output: {response.usage_metadata.candidates_token_count}, "
              f"Total: {response.usage_metadata.total_token_count}")

    line_count = len([l for l in extracted_text.strip().split("\n") if l.strip()])
    print(f"  Extraction complete: {line_count} dialogue lines found.")
    return extracted_text
