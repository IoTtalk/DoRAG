import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_CACHE_DIR = "./models"

_llm = None
_tokenizer = None


def get_model_and_tokenizer():
    global _llm, _tokenizer

    if _llm is None or _tokenizer is None:
        print(f"Loading model: {MODEL_NAME}")
        print(f"Model cache directory: {MODEL_CACHE_DIR}")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR
        )

        _llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda"
        )

        _llm.eval()

    return _llm, _tokenizer


def generate_with_loop(message, histories=None):
    if histories is None:
        histories = []

    history = []

    for human, assistant in histories:
        history.append({"role": "user", "content": human})
        history.append({"role": "assistant", "content": assistant})

    history.append({"role": "user", "content": message})

    model, tokenizer = get_model_and_tokenizer()

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    yield output_text


if __name__ == "__main__":
    user_query = "What is Anthracnose caused by?"

    result = (
        user_query
        + " "
        + "All three species also cause leaf spots "
          "(Howard et al., 1992; Maas and Palm, 1997; Smith, 1998c)."
    )

    histories = []

    generated_answer = generate_with_loop(result, histories)

    answer = ""

    for ans in generated_answer:
        answer = ans

    print(answer)