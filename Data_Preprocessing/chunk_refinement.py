import json
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default configuration
DEFAULT_CHUNK_DIR = "test4"          # Directory containing chunk JSON files to refine
DEFAULT_OUTPUT_DIR = "test5"        # Directory to save refined chunks
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Model to use
DEFAULT_MODELS_DIR = "models"         # Cache directory for models


def create_cleaning_prompt(section_text: str) -> str:
    """Create a prompt to clean section text."""
    return f"""Section to Clean:
{section_text}

Clean the attached Section by correcting any typographical errors and removing all tables, captions, in-text citations (e.g., "({{author name}}, {{year}})") and URLs. "Do not" omit or summarize any content.

Output the cleaned section text only, without any explanation or markdown formatting."""


def generate_refined_section(section_text: str, model, tokenizer, temperature: float = 0.5, max_tokens: int = 4096) -> str:
    """Generate refined section text using HuggingFace model."""
    prompt = create_cleaning_prompt(section_text)
    
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
    
    return answer




def process_files(chunk_dir: str, output_dir: str, 
                  model_name: str = DEFAULT_MODEL_NAME,
                  temperature: float = 0.5, 
                  models_dir: str = DEFAULT_MODELS_DIR):
    """Process all chunk files and refine them using the model."""
    
    chunk_dir = Path(chunk_dir)
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
    
    # Get all JSON files
    chunk_files = sorted([f for f in chunk_dir.glob("*.json") if f.is_file()])
    
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        return
    
    print(f"Found {len(chunk_files)} chunk file(s) to refine\n")
    
    # Process each chunk file
    for chunk_file in chunk_files:
        file_name = chunk_file.name
        print(f"{'='*60}")
        print(f"Processing: {file_name}")
        print(f"{'='*60}")
        
        try:
            # Load chunk file
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            if not isinstance(chunks, list):
                print(f"Error: {file_name} does not contain a list of chunks")
                continue
            
            print(f"Found {len(chunks)} sections to refine")
            
            # Refine each section
            refined_chunks = []
            for idx, section in enumerate(chunks, 1):
                section_title = section.get('section_title', 'Unknown')
                section_text = section.get("content", "")
                
                print(f"\n[{idx}/{len(chunks)}] Refining: {section_title}...")
                
                if not section_text:
                    print(f"Warning: Empty content for section {section_title}")
                    refined_chunks.append(section)
                    continue
                
                try:
                    # Refine the section
                    refined_text = generate_refined_section(section_text, model, tokenizer, temperature=temperature)
                    
                    # Update section with refined content
                    section["content"] = refined_text
                    refined_chunks.append(section)
                    
                    # Preview refined content
                    preview = refined_text[:200]
                    print(f"✓ Refined content preview: {preview}...")
                    
                except Exception as e:
                    print(f"Error refining section {section_title}: {e}")
                    refined_chunks.append(section)
            
            # Save refined chunks
            output_file = output_dir / file_name
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(refined_chunks, f, ensure_ascii=False, indent=4)
                print(f"\n✓ Refined chunks saved to {output_file}")
            except Exception as e:
                print(f"Error saving refined chunks to {output_file}: {e}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
        
        print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine chunk text by removing citations, URLs, and correcting errors"
    )
    parser.add_argument(
        "-c", "--chunk-dir",
        type=str,
        default=DEFAULT_CHUNK_DIR,
        help=f"Directory containing chunk JSON files (default: {DEFAULT_CHUNK_DIR})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save refined chunks (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.5,
        help="Temperature for LLM sampling (default: 0.5)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Cache directory for models (default: {DEFAULT_MODELS_DIR})"
    )
    
    args = parser.parse_args()
    
    process_files(args.chunk_dir, args.output_dir, args.model, args.temperature, args.models_dir)
