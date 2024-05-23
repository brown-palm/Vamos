import json
import random
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, default='gpt4o_result.json', help='Path to the generated answers')
args = parser.parse_args()

def generate_submission(uid_to_answer_map, default_blank_answer):
    """
    Process the uid_to_answer_map:
    1. Replace all blank answers with default_blank_answer.
    2. If subset_ans is True, replaces answers with those from subset_answers (ground truth).

    :param uid_to_answer_map: dict, UID to answer map
    :param default_blank_answer: int, default answer to replace blanks
    :param subset_ans: bool, whether to replace with subset answers or not
    :param subset_answers: dict, subset of answers
    :return: processed submission dictionary
    """

    # Process blank answers in uid_to_answer_map
    for uid, answer in uid_to_answer_map.items():
        if not answer or answer[0] == "":
            uid_to_answer_map[uid] = [str(default_blank_answer)]
        else:
            uid_to_answer_map[uid] = [answer[0]]

    # Convert string answers to integer
    i = 1
    for uid, answer in uid_to_answer_map.items():
      try:
        uid_to_answer_map[uid] = int(answer[0])
        if (int(answer[0]) > 4 or int(answer[0]) < 0):
          i += 1
          uid_to_answer_map[uid] = random.randint(0, 4)
      except: # if there was an invalid answer
        uid_to_answer_map[uid] = random.randint(0, 4)
        i += 1

    return uid_to_answer_map


questions = []
with open('dataset/egoschema_ordered_questions.jsonl', 'r') as file:
  for line in file:
        questions.append(json.loads(line))
        
with open("output/" + args.pred_file, 'r') as file:
    answers = json.load(file)

uid_to_answer = {}
for q, a in zip(questions, answers):
    uid_to_answer[q['q_uid']] = a

# Generate the submission dictionary
submission_dict = generate_submission(uid_to_answer, default_blank_answer=3)
df = pd.DataFrame(submission_dict.items(), columns=['q_uid', 'answer'])
output_file = "output/submission_" + args.pred_file.split('.')[0] + ".csv"
print("save submission file to: ", output_file)
df.to_csv(output_file, index=False)