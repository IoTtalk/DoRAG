import os
import json
import argparse
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(text: str) -> str:
    """Build the extraction prompt"""
    prompt = f"""Paper: 
{text}

Extract structured information of the attached Enterprise Data and return the output in JSON format. Extract the “paper title”, “section titles”, and the “sentence-initial fragment of each section” based on the following extraction rules: 

1. Identify the title of the paper and exclude any author names, metadata. 
2. Identify the main section headings in the paper, typically formatted according to the IMRaD structure or other common academic formats. “Do not” include subsections, figure or table captions. 
3. Include the sentence-initial fragment appearing immediately after each identified section title. Preserve original text, punctuation, and spacing. “Do not” paraphrase or modify.

Output Format (JSON):
{{
  "title": "Extracted Paper Title",
  "sections": [
    {{"section_title": "Section Title 1", "first_words": "First words..."}},
    {{"section_title": "Section Title 2", "first_words": "First words..."}},
    ...
  ]
}}"""
    return prompt


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text that may contain markdown code blocks or extra text"""
    text = text.strip()
    
    # Try to find JSON within markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, try to parse the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"Warning: Could not extract valid JSON from: {text[:100]}...")
        return {"title": "", "sections": []}


def generate_answer(text: str, model, tokenizer, temperature: float = 0.7, max_tokens: int = 4096) -> dict:
    """Generate answer using HuggingFace model"""
    prompt = build_prompt(text)
    
    # Prepare input
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate
    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the output
    answer = answer.replace(tokenizer.decode(inputs[0], skip_special_tokens=True), "").strip()
    
    # Extract JSON from the answer
    return extract_json_from_text(answer)


def process_files(input_dir: str, output_dir: str, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
                  temperature: float = 0.7, done_file: str = None, models_dir: str = "models"):
    """Process all text files in input directory"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    print(f"Model cache directory: {models_dir.absolute()}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(models_dir)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(models_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load done files if specified
    done_files = set()
    if done_file and Path(done_file).exists():
        with open(done_file, 'r') as f:
            done_files = set(line.strip() for line in f.readlines())
    
    # Get all text files
    text_files = sorted([f for f in input_dir.glob("*.txt") if f.is_file()])
    print(f"Found {len(text_files)} files to process")
    
    processed_count = 0
    
    for file_path in text_files:
        if file_path.name in done_files:
            print(f"Skipping {file_path.name} (already processed)")
            continue
        
        try:
            print(f"\nProcessing: {file_path.name}")
            
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            char_length = len(text)
            word_length = len(text.split())
            print(f"  Length: {char_length} chars, {word_length} words")
            
            # Generate answer
            answer = generate_answer(text, model, tokenizer, temperature=temperature)
            
            # Save result
            output_file = output_dir / f"{file_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": file_path.name,
                    "extraction": answer
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  Saved to: {output_file}")
            
            processed_count += 1
            
            # Update done file
            if done_file:
                with open(done_file, 'a', encoding='utf-8') as f:
                    f.write(f"{file_path.name}\n")
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
    
    print(f"\n\nProcessed {processed_count} files successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured information from papers")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing text files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON outputs')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='HuggingFace model name')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--done_file', type=str, default=None, help='File to track processed files')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to cache models')
    
    args = parser.parse_args()
    
    process_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        temperature=args.temperature,
        done_file=args.done_file,
        models_dir=args.models_dir
    )
    
