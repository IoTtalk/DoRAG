from utils.use_ollama import query_ollama
import os
import json
import re
import argparse


# Dataset generation parameters
query_type_options = ["common", "niche", "highly specialized"]
query_format_options = ["open-ended", "yes-no"]
query_length_options = ["less than 15 words", "more than 15 words"]
doc_words_options = [50, 100, 150]


def get_finetuning_prompt(positive_document, query_type, query_format, query_length, doc_words):
    """Generate fine-tuning dataset prompt"""

    prompt = f"""Based on the positive document below, generate:
1. A {query_type}, {query_format}, {query_length} query that the document answers.
2. A hard negative document appears topically similar to the query by sharing related keywords or concepts, but it does not actually answer the query or match its intent. It is at least {doc_words} words long.

Output Format:
{{
  "query": "...",
  "positive_document": "...",
  "hard_negative_document": "..."
}}

Document:
{positive_document}

IMPORTANT: Output ONLY the JSON object, nothing else."""
    return prompt


def extract_json_from_response(response):
    """Extract JSON object from LLM response"""

    if not response:
        return None

    # Remove possible markdown code block
    response = response.strip()
    response = re.sub(r"^```json", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"^```", "", response).strip()
    response = re.sub(r"```$", "", response).strip()

    # Try direct JSON parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from response
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    return None


def get_finetuning_data_from_deepseek(prompt):
    """Generate fine-tuning data through Deepseek API"""

    response = query_ollama(prompt, model="deepseek-r1:70b")

    data = extract_json_from_response(response)

    if not data:
        return None

    required_keys = ["query", "positive_document", "hard_negative_document"]

    if all(key in data for key in required_keys):
        return data

    return None


def main():
    """Read chunk.json and generate fine-tuning data"""

    parser = argparse.ArgumentParser(
        description="Generate fine-tuning dataset from chunk.json"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="chunk.json",
        help="Path to input JSON file. Default: chunk.json"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="finetuning_dataset.json",
        help="Path to output JSON file. Default: finetuning_dataset.json"
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Maximum retry attempts for each prompt. Default: 3"
    )

    args = parser.parse_args()

    # Read input file
    print(f"Reading input file: {args.input}")

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON format: {args.input}")
        return

    print(f"Successfully read {len(data)} chunks")

    # Create output directory
    output_dir = os.path.dirname(args.output)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = []
    success_count = 0
    fail_count = 0
    skipped_count = 0

    total_prompt_count = (
        len(data)
        * len(query_type_options)
        * len(query_format_options)
        * len(query_length_options)
        * len(doc_words_options)
    )

    print(f"Expected maximum generated samples: {total_prompt_count}")
    print("=" * 60)

    for idx, entry in enumerate(data):
        # Extract content from chunk.json
        positive_document = entry.get("content", "")

        if not positive_document:
            print(f"[WARN] Chunk {idx} has no content, skipping")
            skipped_count += 1
            continue

        positive_document = positive_document.strip()

        # chunk_id: use original id if exists, otherwise use index
        chunk_id = entry.get("id", idx)

        for q_type in query_type_options:
            for q_format in query_format_options:
                for q_length in query_length_options:
                    for doc_words in doc_words_options:

                        prompt = get_finetuning_prompt(
                            positive_document=positive_document,
                            query_type=q_type,
                            query_format=q_format,
                            query_length=q_length,
                            doc_words=doc_words
                        )

                        print(
                            f"Processing chunk {chunk_id} | "
                            f"{q_type}, {q_format}, {q_length}, {doc_words} words...",
                            end=" "
                        )

                        attempt = 0
                        generated_data = None

                        while attempt < args.max_attempts and generated_data is None:
                            attempt += 1

                            generated_data = get_finetuning_data_from_deepseek(prompt)

                            if generated_data is None:
                                print(f"[Attempt {attempt}/{args.max_attempts} failed]", end=" ")                            

                        if generated_data:
                            output_item = {
                                "query": generated_data["query"],
                                "id": len(all_data),
                                "chunk_id": chunk_id,
                                "positive_doc": positive_document,
                                "hard_negative_doc": generated_data["hard_negative_document"]
                            }
                            print(output_item)

                            all_data.append(output_item)
                            success_count += 1

                            print("✓ Success")
                        else:
                            fail_count += 1
                            print("✗ Failed")

    # Save data to JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print()
    print("=" * 60)
    print("Done!")
    print(f"Successfully generated: {success_count} samples")
    print(f"Failed: {fail_count} samples")
    print(f"Skipped chunks: {skipped_count}")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()