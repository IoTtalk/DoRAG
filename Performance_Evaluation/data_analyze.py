import json
import statistics
import math
import random

N = 1000
randn = -1  # Set to -1 to use all data, or specify a number for random sampling

# Load the JSON file
file_path = 'test7\\1292_k5_vs_k10_tart_stella_llama32.json'  # Replace with the actual file path
with open(file_path, 'r', encoding="utf-8") as file:
    data = json.load(file)

if randn < 0:
    randn = len(data)

sampled_entries = random.sample(data, randn)
# Extract scores for each field
answer_1_scores = [entry['answer_1_score'] for entry in sampled_entries if 'answer_1_score' in entry]
answer_2_scores = [entry['answer_2_score'] for entry in sampled_entries if 'answer_2_score' in entry]

# Calculate the mean and standard deviation for each field
results = {    
    "answer_1_score": {
        "mean": statistics.mean(answer_1_scores),
        "std_dev": statistics.stdev(answer_1_scores)
    },
    "answer_2_score": {
        "mean": statistics.mean(answer_2_scores),
        "std_dev": statistics.stdev(answer_2_scores)
    }
}

answer_win, answer_lose, answer_draw = 0, 0, 0
answer_win_qids, answer_lose_qids, answer_draw_qids = [], [], []

same_context_count = 0
same_context_qids = []

for entry in sampled_entries:
    qid = entry.get('qid', None)  # Get the qid if available
    
    if 'answer_1_score' in entry and 'answer_2_score' in entry:
        if entry['answer_1_score'] > entry['answer_2_score']:
            answer_win += 1
            answer_win_qids.append(qid)
        elif entry['answer_1_score'] < entry['answer_2_score']:
            answer_lose += 1
            answer_lose_qids.append(qid)
        else:
            answer_draw += 1
            answer_draw_qids.append(qid)

    # Compare strings in 'context_1' and 'context_2'
    if 'retrieved_context_1' in entry and 'retrieved_context_2' in entry and entry['retrieved_context_1'] == entry['retrieved_context_2']:
        same_context_count += 1
        same_context_qids.append(qid)

n = len(sampled_entries)  # Sample size (same as total comparisons here)

win_rate_1 = answer_win / n
win_rate_2 = answer_lose / n

# Confidence level (95%)
z_score = 1.96

# Calculate standard errors
se_1 = math.sqrt(win_rate_1 * (1 - win_rate_1) / n)
se_2 = math.sqrt(win_rate_2 * (1 - win_rate_2) / n)

# Apply Finite Population Correction (FPC)
fpc = math.sqrt((N - n) / (N - 1))
se_1_fpc = se_1 * fpc
se_2_fpc = se_2 * fpc

# Compute confidence intervals with FPC
ci_1_lower = win_rate_1 - z_score * se_1_fpc
ci_1_upper = win_rate_1 + z_score * se_1_fpc

ci_2_lower = win_rate_2 - z_score * se_2_fpc
ci_2_upper = win_rate_2 + z_score * se_2_fpc

# Print the results
print("Mean and Standard Deviation for Scores:")
for score_type, stats in results.items():
    print(f"{score_type} - Mean: {stats['mean']:.2f}, Standard Deviation: {stats['std_dev']:.2f}")

print("\nComparison Results (answer_1_score vs. answer_2_score):")
print(f"Win: {answer_win} times, QIDs: {answer_win_qids}")
print(f"Lose: {answer_lose} times, QIDs: {answer_lose_qids}")
print(f"Draw: {answer_draw} times, QIDs: {answer_draw_qids}")

print("\nWin Rates and Confidence Intervals (with FPC):")
print(f"Answer 1 Win Rate: {win_rate_1:.2%}")
print(f"95% Confidence Interval for Answer 1: ({ci_1_lower:.2%}, {ci_1_upper:.2%})")
print(f"Answer 2 Win Rate: {win_rate_2:.2%}")
print(f"95% Confidence Interval for Answer 2: ({ci_2_lower:.2%}, {ci_2_upper:.2%})")

# Path to save the results file
output_file_path = file_path[:-4] + 'analyze_result_' + str(len(answer_1_scores)) + '.txt'

# Open the file in write mode
with open(output_file_path, 'w') as output_file:
    # Write mean and standard deviation
    output_file.write("Mean and Standard Deviation for Scores:\n")
    for score_type, stats in results.items():
        output_file.write(f"{score_type} - Mean: {stats['mean']}, Standard Deviation: {stats['std_dev']}\n") 
    
    # Write answer comparison results
    output_file.write("\nComparison Results (answer_1_score vs. answer_2_score):\n")
    output_file.write(f"Win: {answer_win} times, QIDs: {answer_win_qids}\n")
    output_file.write(f"Lose: {answer_lose} times, QIDs: {answer_lose_qids}\n")
    output_file.write(f"Draw: {answer_draw} times, QIDs: {answer_draw_qids}\n")
    
    output_file.write("\nWin Rates and Confidence Intervals (with FPC):\n")
    output_file.write(f"Answer 1 Win Rate: {win_rate_1:.2%}\n")
    output_file.write(f"95% Confidence Interval for Answer 1: ({ci_1_lower:.2%}, {ci_1_upper:.2%})\n")
    output_file.write(f"Answer 2 Win Rate: {win_rate_2:.2%}\n")
    output_file.write(f"95% Confidence Interval for Answer 2: ({ci_2_lower:.2%}, {ci_2_upper:.2%})\n")

print(f"Analysis results have been saved to {output_file_path}")

