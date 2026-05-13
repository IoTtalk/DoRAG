import json
import os
import re
import argparse
from typing import Union, Tuple
from utils.use_ollama import query_ollama

# 解析模型回覆
def parse_eval(response: Union[str, list]) -> Tuple[bool, bool, str]:
    if isinstance(response, list):
        text = " ".join(response)
    else:
        text = response

    pos_match = re.search(
        r"positive\s*document\s*:\s*\{*\s*(true|false)\s*\}*", text, re.IGNORECASE
    )
    neg_match = re.search(
        r"(hard[\s-]*negative)\s*document\s*:\s*\{*\s*(true|false)\s*\}*", text, re.IGNORECASE
    )
    expl_match = re.search(
        r"explanation\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL
    )

    pos_check = (pos_match.group(1).lower() == "true") if pos_match else None
    neg_check = (neg_match.group(2).lower() == "true") if neg_match else None
    explanation = expl_match.group(1).strip() if expl_match else None

    return pos_check, neg_check, explanation


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()


# 檔案路徑
input_file = args.input_file
output_file = args.output_file

# 讀取初始資料
with open(input_file, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 如果已存在結果檔，接續進度
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"檢測到已存在的結果檔，繼續處理未完成部分...")
else:
    data = original_data  # 從頭開始

# 找出已完成進度
start_idx = 0
for i, item in enumerate(data):
    if "positive_doc_check_llama33" not in item or "hard_negative_doc_check_llama33" not in item:
        start_idx = i
        break
else:
    print("所有項目都已完成！")
    exit()

print(f"從第 {start_idx+1} 筆開始處理，共 {len(data)} 筆")

# 處理剩下的
for idx in range(start_idx, len(data)):
    item = data[idx]
    question = item["query"]
    positive_doc = item["positive_doc"]
    hard_negative_doc = item["hard_negative_doc"]

    prompt = (
        f"You are given a query, a positive document, and a hard negative document:\n"
        f"1. Verify that the positive document is highly relevant to and directly useful for answering the query.\n"
        f"2. Verify that the hard negative document is not relevant to answering the query but remains topically related.\n\n"
        f"Query: {question}\n"
        f"Positive Document: {positive_doc}\n"
        f"Hard Negative Document: {hard_negative_doc}\n"
        f"Here is an example output: Positive Document: {{true or false}}\n"
        f"Hard negative Document: {{true or false}}\n"
        f"Explanation:"
    )

    model_response = query_ollama(prompt, model="llama3.3")
    pos_check, neg_check, explanation = parse_eval(model_response)

    # 安全檢查，無法解析就直接退出
    if pos_check is None or neg_check is None:
        exit(1)

    # 新增模型檢查結果欄位
    item["positive_doc_check_llama33"] = pos_check
    item["hard_negative_doc_check_llama33"] = neg_check
    item["explanation_llama33"] = explanation

    # 即時存檔
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[{idx+1}/{len(data)}] 已處理 chunk_id={item.get('chunk_id')}")