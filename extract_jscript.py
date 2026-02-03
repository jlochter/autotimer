from pdf2image import convert_from_path
import argparse
import os
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

def extract_jscript(pdf_path, output_path, limit_pages=None, api_key=None):
    """
    Extracts text from a Japanese PDF script using Gemini 2.5 Flash Lite.
    Rotates images 270 degrees before sending to Gemini.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the extracted text.
        limit_pages (int): Optional limit on number of pages to process.
        api_key (str): Gemini API Key.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments.")

    client = genai.Client(api_key=api_key)
    model_id = 'gemini-2.5-flash-lite' 

    print(f"Converting PDF to images: {pdf_path}...")
    if limit_pages:
        images = convert_from_path(pdf_path, last_page=limit_pages, dpi=300)
    else:
        images = convert_from_path(pdf_path, dpi=300)
    
    full_text = []
    total_tokens = 0
    token_usage_list = []
    last_line = None
    
    # Initialize the file to clear it
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    for i, image in enumerate(tqdm(images, desc="Processing pages", unit="page")):
        # Rotate image 270 degrees as requested
        rotated_image = image.rotate(270, expand=True)
        
        # Convert PIL to bytes for Gemini
        img_byte_arr = io.BytesIO()
        rotated_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Send to Gemini with basic retry
        success = False
        retries = 3
        page_raw_text = ""
        
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[
                        "This image is a page from a Japanese anime/game script. Only extract text if you find a script table. For each dialogue entry in the table, output strictly in the format 'actor : dialogue', with exactly one entry per line. Do not include scene descriptions, JSON, bounding boxes, or any introductory text. If no script table or dialogue is found, return an empty string.",
                        types.Part.from_bytes(data=img_bytes, mime_type='image/png')
                    ],
                    config=types.GenerateContentConfig(
                        max_output_tokens=1500,
                        temperature=0.0
                    )
                )
                
                if response.usage_metadata:
                    page_tokens = response.usage_metadata.total_token_count
                    total_tokens += page_tokens
                    token_usage_list.append(page_tokens)

                page_raw_text = response.text or ""
                success = True
                break
            except Exception as e:
                print(f"\nError on page {i+1}, attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    page_raw_text = f"\n[ERROR: Failed to process page {i+1}]\n"
        
        # Deduplicate consecutive lines within the page and across pages
        lines = [line.strip() for line in page_raw_text.splitlines() if line.strip()]
        dedup_lines = []
        for line in lines:
            if line != last_line:
                dedup_lines.append(line)
                last_line = line
        
        page_clean_text = "\n".join(dedup_lines)
        if page_clean_text:
            full_text.append(page_clean_text)
            # Append to file progressively
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(page_clean_text + "\n")
        
    extracted_text = "\n".join(full_text)
    
    print("\n--- OCR Token Usage Summary ---")
    print(f"Total tokens used: {total_tokens}")
    if token_usage_list:
        avg_tokens = total_tokens / len(token_usage_list)
        print(f"Average tokens per page: {avg_tokens:.2f}")
    print("--------------------------------\n")

    print(f"Final extracted text saved to {output_path}.")
    return extracted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Japanese text from PDF script using Gemini.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--output", required=True, help="Path to output text file")
    parser.add_argument("--limit_pages", type=int, help="Limit number of pages to process")
    parser.add_argument("--api_key", help="Gemini API Key")
    
    args = parser.parse_args()
    
    extract_jscript(args.pdf, args.output, args.limit_pages, args.api_key)
