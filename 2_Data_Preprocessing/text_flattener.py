import os
import glob
import argparse

parser = argparse.ArgumentParser(description='Flatten extracted text files')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing extracted text files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save flattened text')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all text files in the directory
file_names = glob.glob(os.path.join(input_dir, "*.txt"))

for txt_path in file_names:
    txt_name = os.path.basename(txt_path)  # Extract filename
    text_output_path = os.path.join(output_dir, txt_name)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
        plain_text = ' '.join(text.split()).replace("- ", "-")    
    
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(plain_text)
        print(f"Plain text saved to: {text_output_path}")