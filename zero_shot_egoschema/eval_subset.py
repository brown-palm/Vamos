import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, default='gpt4o_result.json', help='Path to the generated answers')
args = parser.parse_args()

questions = []
with open('dataset/egoschema_ordered_questions.jsonl', 'r') as file:
  for line in file:
        questions.append(json.loads(line))

with open('dataset/subset_answers.json', 'r') as file:
    subset_answers = json.load(file)
    
with open("output/" + args.pred_file, 'r') as file:
    answers = json.load(file)

uid_to_answer = {}
for q, a in zip(questions, answers):
    uid_to_answer[q['q_uid']] = a

print(len(uid_to_answer))

wrong_count = 0
matched_count = 0
no_match = 0
blank_count = 0
few_shot = 0

for uid in subset_answers:
    subset_answer = subset_answers[uid]
    generated_answer = uid_to_answer.get(uid)

    if generated_answer is None :
        print(f"No answer found for UID: {uid}")
        no_match += 1
    elif len(generated_answer[0]) > 1:
        print(f"No answeasdID: {uid}")
        wrong_count += 1
    elif not generated_answer or generated_answer[0] == "":
        print(f"Wrong (blank answer) for UID: {uid}")
        # wrong_count += 1
        blank_count += 1
    elif int(generated_answer[0]) != subset_answer:
        print(f"Discrepancy for UID: {uid}, Subset: {subset_answer}, Generated: {generated_answer[0]}")
        wrong_count += 1
    else:
        print(f"Matching answer for UID: {uid}")
        matched_count += 1

print("---------------------------------------------------------")
print(f"Total UIDs checked: {len(subset_answers)}")
print(f"Total matched answers: {matched_count}")
print(f"Total blank answers: {blank_count}")
print(f"Total wrong answers: {wrong_count}")

print("subset accuracy: ", matched_count / len(subset_answers))

