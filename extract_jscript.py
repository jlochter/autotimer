import easyocr
from pdf2image import convert_from_path
import argparse
import os
import numpy as np

def extract_jscript(pdf_path, output_path):
    """
    Extracts text from a Japanese PDF script using OCR.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the extracted text.
    """
    print(f"Converting PDF to images: {pdf_path}...")
    # Convert PDF to list of images (one per page)
    images = convert_from_path(pdf_path)
    
    print(f"Initializing EasyOCR for Japanese...")
    reader = easyocr.Reader(['ja'], gpu=False) # Set gpu=True if CUDA is available, but Mac usually uses CPU/MPS for this via easyocr logic or cpu fallback
    
    full_text = []
    
    for i, image in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}...")
        # EasyOCR expects a file path, numpy array, or bytes. 
        # pdf2image returns PIL images, so we convert to numpy array.
        image_np = np.array(image)
        
        # detail=0 returns just the text list
        result = reader.readtext(image_np, detail=0) 
        
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
    
    args = parser.parse_args()
    
    extract_jscript(args.pdf, args.output)
