import utils.call_gpt as call_gpt
import os
import json
from tqdm import tqdm
import random
import re

N = 5
n = 5

#random.seed(1023)
rand_idx = random.sample(range(N), n)
print(f"Selected random indices: {rand_idx}")

# 路徑設置
file1 = "test6/testing.json"
file2 = "test6/testing2.json"
header1 = "answer_5"
header2 = "answer_1" 
output_dir = "test7"
output_file = os.path.join(output_dir, "tt.json")
question_file = "Qs.txt"

class ScoreParseError(Exception):
    pass

def extract_score(score_line):
    try:
        score_part = score_line.split(': ')[1].strip()
        return int(score_part.split('/')[0])
    except (IndexError, ValueError) as e:
        raise ScoreParseError(f"Failed to extract score from line: '{score_line}'") from e


os.makedirs(output_dir, exist_ok=True)

with open(question_file, 'r', encoding='utf-8') as qfr:
    question_list = [line.strip() for line in qfr.readlines()]

def read_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

data_file1 = read_json_data(file1)
data_file2 = read_json_data(file2)

answers_1 = [item[header1].replace('\n', ' ') for item in data_file1]
answers_2 = [item[header2].replace('\n', ' ') for item in data_file2]

print(f"Questions: {len(question_list)}")
print(f"Answers from file1: {len(answers_1)}")
print(f"Answers from file2: {len(answers_2)}")

init = 0
done_qid = []
data = []

if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        data = json.load(f)
    done_qid = [item["qid"] for item in data]

print(f"The initialized index is: {init}")

for i in tqdm(rand_idx, desc="Processing questions"):
    if i not in done_qid:
        tqdm.write(str(i) + ":")

        ans1 = answers_1[i]
        ans2 = answers_2[i]

        # 隨機打亂順序：(原始 index, 答案內容)
        shuffled = [(1, ans1), (2, ans2)]
        random.shuffle(shuffled)

        # Debug：顯示目前傳入順序與 preview
        print(f"[Debug] QID {i}: First = Answer {shuffled[0][0]} ({shuffled[0][1][:50]}...), "
              f"Second = Answer {shuffled[1][0]} ({shuffled[1][1][:50]}...)")

        # 呼叫 GPT，傳入亂序答案
        response = call_gpt.ask_gpt_for_final_answer(
            question_list[i],
            shuffled[0][1],
            shuffled[1][1]
        )

        # 處理回傳格式
        if "Answer 2" in response[0] or "Answer2" in response[0]:
            answer_score = response[0].split('\n')
            feedback = response[1:]
        else:
            answer_score = response[0:2]
            feedback = response[2:]

        # GPT 評分根據 shuffled 順序，對應回原始 Answer 1/2
        gpt_score = {
            "Answer 1": extract_score(answer_score[0]),
            "Answer 2": extract_score(answer_score[1])
        }

        # score_map: 將 GPT 評分對回原始的答案順序
        score_map = {
            shuffled[0][0]: gpt_score["Answer 1"],
            shuffled[1][0]: gpt_score["Answer 2"]
        }

        new_entry = {
            'qid': i,
            'query': question_list[i],
            'answer_1_score': score_map[1],
            'answer_2_score': score_map[2],
            'feedback': ' '.join(feedback),
            'answer_1': ans1,
            'answer_2': ans2
        }

        data.append(new_entry)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
