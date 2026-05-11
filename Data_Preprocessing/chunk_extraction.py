import re
import json
import os
import argparse
from pathlib import Path


# Default configuration
DEFAULT_IDENTIFIER_DIR = "test3"        # Directory containing identifier JSON files
DEFAULT_TEXT_DIR = "test2"              # Directory containing text files to process
DEFAULT_OUTPUT_DIR = "test4"           # Directory to save extracted chunks


def load_identifier(identifier_path):
    """Load identifier configuration from JSON file."""
    with open(identifier_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_text(text, section):
    """Split text based on section identifier configuration."""
    section_title = re.escape(section["section_title"])
    first_words = re.escape(section["first_words"])

    matches_first_words = list(re.finditer(first_words, text, re.IGNORECASE))
    matches_section_title = list(re.finditer(section_title, text, re.IGNORECASE))

    split_index = None

    for match_first in matches_first_words:
        for match_title in matches_section_title:
            if 0 < match_first.start() - match_title.end() < 128:
                split_index = match_title.start()
                print(f"[{section['section_title']}] Match found: section_title at {match_title.start()} and first_words at {match_first.start()}")
                break
        if split_index is not None:
            break

    if split_index is None:
        if matches_first_words and ("abstract" in section_title.lower() or "introduction" in section_title.lower()):
            split_index = matches_first_words[0].start()
            print(f"[{section['section_title']}] Using first occurrence of first_words at index: {split_index}")

    if split_index is not None:
        part1, part2 = text[:split_index], text[split_index:]
        print(f"[{section['section_title']}] Splitting text at index: {split_index}")

        # Remove the section title from beginning of part2
        clean_start = re.escape(section["section_title"])
        part2 = re.sub(rf"^{clean_start}\s*", "", part2, flags=re.IGNORECASE)
        if re.match(r"^[A-Z0-9(]", part2):
            return part1, part2, section["section_title"]
        else:
            return None, text, None
    else:
        print(f"[{section['section_title']}] No valid split found.")
        return None, text, None


def split_into_sections(text, section_information):
    """Split text into sections based on identifier configuration."""
    section_list = []
    
    # Handle both nested and flat structures
    if "extraction" in section_information:
        extraction = section_information["extraction"]
        title = extraction.get("title", "Unknown")
        sections = extraction.get("sections", [])
    else:
        title = section_information.get("title", "Unknown")
        sections = section_information.get("sections", [])
    
    index = 0

    previous_title = None
    while index < len(sections):
        section = sections[index]
        former_text, text, current_title = split_text(text, section)

        if former_text is not None:
            # Save the previous title with the current chunk
            section_list.append({
                "title": title,
                "section_title": previous_title,
                "content": former_text
            })            
            previous_title = current_title  # Save current title for next chunk

        index += 1

    # Append the final remaining section with the last known section title
    section_list.append({
        "title": title,
        "section_title": previous_title or sections[-1]["section_title"] if sections else "Unknown",
        "content": text
    })

    return section_list


def process_files(identifier_dir, text_dir, output_dir):
    """Process all text files using corresponding identifiers."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of identifier files
    identifier_files = list(Path(identifier_dir).glob("*.json"))
    
    if not identifier_files:
        print(f"No identifier files found in {identifier_dir}")
        return
    
    print(f"Found {len(identifier_files)} identifier file(s)")
    
    # Process each identifier
    for identifier_path in identifier_files:
        identifier_name = identifier_path.stem  # filename without extension
        
        # Load identifier configuration
        try:
            section_information = load_identifier(identifier_path)
        except Exception as e:
            print(f"Error loading identifier {identifier_path}: {e}")
            continue
        
        # Find corresponding text file
        text_file_path = Path(text_dir) / f"{identifier_name}.txt"
        
        if not text_file_path.exists():
            print(f"Text file not found: {text_file_path}")
            continue
        
        # Read text file
        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading text file {text_file_path}: {e}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {identifier_name}")
        print(f"{'='*60}")
        
        # Split text into sections
        section_list = split_into_sections(text, section_information)
        
        # Display results
        for section in section_list:
            print(f"Title: {section['title']}")
            print(f"Section Title: {section['section_title']}")
            preview = section["content"][:500]
            print(f"Content preview: {preview}...")
            print(f"Char Length: {len(section['content'])}   Word Length: {len(section['content'].split())}")
            print("-" * 60)
        
        # Save to output directory
        output_path = Path(output_dir) / f"{identifier_name}.json"
        try:
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(section_list, json_file, ensure_ascii=False, indent=4)
            print(f"✓ Sections saved to {output_path}")
        except Exception as e:
            print(f"Error saving sections to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text files and split them into sections using identifiers"
    )
    parser.add_argument(
        "-i", "--identifier-dir",
        type=str,
        default=DEFAULT_IDENTIFIER_DIR,
        help=f"Directory containing identifier JSON files (default: {DEFAULT_IDENTIFIER_DIR})"
    )
    parser.add_argument(
        "-t", "--text-dir",
        type=str,
        default=DEFAULT_TEXT_DIR,
        help=f"Directory containing text files to process (default: {DEFAULT_TEXT_DIR})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save extracted chunks (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    process_files(args.identifier_dir, args.text_dir, args.output_dir)

