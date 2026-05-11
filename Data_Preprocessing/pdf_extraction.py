import os
import glob
import argparse
import fitz  # PyMuPDF

parser = argparse.ArgumentParser(description='Extract text from PDF files')
parser.add_argument('--pdf_dir', type=str, required=True, help='Directory containing PDF files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted text')
args = parser.parse_args()

pdf_dir = args.pdf_dir
output_dir = args.output_dir

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all PDF files in the directory
file_names = glob.glob(os.path.join(pdf_dir, "*.pdf"))

for pdf_path in file_names:
    pdf_name = os.path.basename(pdf_path)  # Extract filename
    text_output_path = os.path.join(output_dir, pdf_name.replace(".pdf", ".txt"))
    
    with fitz.open(pdf_path) as doc:
        text_fitz = "".join(page.get_text() for page in doc)
    
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(text_fitz)
    print(f"Extracted text saved to: {text_output_path}")