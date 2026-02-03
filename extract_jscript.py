import easyocr
from pdf2image import convert_from_path
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

def extract_jscript(pdf_path, output_path, limit_pages=None):
    """
    Extracts text from a Japanese PDF script using OCR.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the extracted text.
        limit_pages (int): Optional limit on number of pages to process.
    """
    print(f"Converting PDF to images: {pdf_path}...")
    # Convert PDF to list of images (one per page)
    # If limit is set, we can optimize by only converting what we need, 
    # but convert_from_path loads all by default unless we use first_page/last_page
    # simpler to just slice list if not optimizing memory heavily yet
    # Actually pdf2image supports first_page and last_page
    if limit_pages:
        images = convert_from_path(pdf_path, last_page=limit_pages, dpi=300)
    else:
        images = convert_from_path(pdf_path, dpi=300)
    
    print(f"Initializing EasyOCR for Japanese...")
    reader = easyocr.Reader(['ja'], gpu=False) # Set gpu=True if CUDA is available, but Mac usually uses CPU/MPS for this via easyocr logic or cpu fallback
    
    full_text = []
    
    for image in tqdm(images, desc="Processing pages", unit="page"):
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Preprocessing with OpenCV
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 2. Apply thresholding (Otsu's binarization) to get black text on white background
        # This helps significantly with noise and contrast
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # EasyOCR works on numpy arrays
        # detail=0 returns just the text list
        result = reader.readtext(binary, detail=0) 
        
        page_text = "\n".join(result)
        full_text.append(page_text)
        
    extracted_text = "\n\n".join(full_text)
    
    print(f"Saving extracted text to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
        
    return extracted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Japanese text from PDF script.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--output", required=True, help="Path to output text file")
    
    parser.add_argument("--limit_pages", type=int, help="Limit number of pages to process (for testing)")
    
    args = parser.parse_args()
    
    extract_jscript(args.pdf, args.output, args.limit_pages)
