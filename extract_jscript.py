import argparse
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def extract_jscript(pdf_path, output_path, api_key=None):
    """
    Extracts text from a Japanese PDF script using Gemini 2.5 Flash.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the extracted text.
        api_key (str): Gemini API Key.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments.")

    client = genai.Client(api_key=api_key)
    model_id = 'gemini-2.5-flash' 

    print(f"Reading PDF file: {pdf_path}...")
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    prompt = """
Read the PDF file (Japanese animation script (Tategaki)) and extract all dialogues, including actor and text.

Directionality: The reading order is Vertical (top-to-bottom) and moves Right-to-Left across the page.
Structure: Treat the vertical lines of text on the far right as the beginning (Column 1). Move leftward for subsequent dialogue/columns.

Task: Extract the text and output it as a list where first item corresponds to the far-right vertical column.

Output each dialogue in a line, following the format:
ACTOR:TEXT

Do not output comments or anything else.
"""

    print(f"Sending PDF to Gemini {model_id}...")
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(
                data=pdf_data,
                mime_type='application/pdf'
            ),
            prompt
        ]
    )
    
    extracted_text = response.text or ""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print(f"Final extracted text saved to {output_path}.")
    return extracted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Japanese text from PDF script using Gemini.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--output", required=True, help="Path to output text file")
    parser.add_argument("--api_key", help="Gemini API Key")
    
    args = parser.parse_args()
    
    extract_jscript(args.pdf, args.output, args.api_key)
