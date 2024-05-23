from openai import OpenAI
import json
import os
import pandas as pd
import numpy as np
import time
import argparse
import numpy as np
import re

sys_message = [{
    'role': 'system',
    'content': f"Suppose we have narrations of a video describing what a person is doing. We would like to answer the choose the right option number to answer the question correctly based on the information. Even if there is a lack of information, you HAVE to choose what you think is likely most correct. You always have to output an option number.",
    }]

demos = [
    # 0074f737-11cb-497d-8d07-77c3a8127391
    "Narrations: a person typing on a laptop in a living room; a person using a laptop on a table; a person using a laptop on a table; a person using a laptop in a living room; a person using a laptop on a table; a person using a laptop on a table; a person using a laptop on a table; a person using a laptop in a living room; a person using a laptop on a table in a living room; a person using a laptop on a table in a living room; a person using a laptop on a table in a living room; a person using a laptop on a table in a living room; a person using a laptop in a living room; a person using a laptop on a table in a living room; a person using a laptop on a table in a living room; a person using a laptop on a table in a living room; a person using a laptop on a table in a living room; a person using a laptop in a living room; a person using a laptop in a living room; a person using a laptop on a table; a person using a laptop on a table; a person using a laptop in a living room; a person using a laptop in a living room; a person using a laptop at a table; Question: Taking into account all the actions performed by c, what can you deduce about the primary objective and focus within the video content?; Answer Options:\n option 0. C is cooking.\n option 1. C is doing laundry.\n option 2. C is cleaning the kitchen.\n option 3. C is cleaning dishes.\n option 4. C is cleaning the bathroom.;\n Correct Answer: 3 ###\n",
    # 00b9a0de-c59e-49cb-a127-6081e2fb8c8e
    "Narrations: a person is painting on a table with an ipad; a person holding a piece of paper; a woman is painting a picture of a woman and a man; a person drawing on a piece of paper; a person drawing on a piece of paper; a person drawing a picture of a man and a woman; a person is holding a piece of paper with a drawing on it; a person drawing a picture on a piece of paper; a person is drawing a picture on a piece of paper; a person drawing with a pencil on a table; a person drawing a picture on a piece of paper; a person is drawing a picture of a man and a woman; a person holding a piece of paper with a drawing on it; a person is drawing a picture on a piece of paper; a person is drawing on a piece of paper; a person is holding a paint brush; a woman is painting a portrait of a man and a woman; a person is drawing a picture of a man and a woman; a person is drawing a picture on a piece of paper; a person is drawing a picture of three people on a piece of paper; a person drawing a picture on a piece of paper; a person is painting a picture on a table; a person drawing a picture on a tablet computer; a person drawing on a piece of paper; Question: What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?; Answer Options:\n option 0. To provide a source of water for the paintbrush.\n option 1. To provide a place to store the paintbrush.\n option 2. To provide a place to dispose of the paintbrush.\n option 3. To provide a place to rest the paintbrush.\n option 4. To clean the paintbrush.;\n Correct answer: 4 ###\n",
    # 011b8b73-0ce4-4843-95ef-33b79610d212
    "Narrations: a man standing next to a water heater; a person walking on the grass next to a van; a man putting a white pipe into a trash can; a person standing on a lawn mower with a white blade; a person cutting a long piece of white paper; a man using a circular saw on a lawn; a man using a circular saw on a lawn; a person is standing on the steps of a house; a person holding a wooden stick in a living room; a man is using a piece of wood to make a floor; a man is working on a wooden floor; a man is working on a wooden floor in a room; a man is working on a ladder in a backyard; a person standing on a lawn mower in front of a house; a man using a saw to cut a piece of wood; a man using a circular saw to cut a piece of wood; a man using a circular saw to cut a piece of wood; a man using a saw to cut a piece of wood; a person using a circular saw to cut a piece of wood; a man is working on the steps of a house; a man is cleaning a room with a vacuum; a man is working on a wooden floor in a room; a man is working on a corner of a room; a man is working on a piece of wood in a room; Question: What can be deduced about c's level of expertise in the task by observing the kind of adjustments made throughout the video? ; Answer Options:\n option 0. C is a novice woodworker. he was not able to cut the wood to size and install it on the wall without making several adjustments.\n option 1. C is an expert woodworker. he was able to cut the wood to size and install it on the wall without making any adjustments.\n option 2. C is a professional woodworker. he was able to cut the wood to size and install it on the wall in a timely and efficient manner.\n option 3. C is an experienced woodworker. he was able to cut the wood to size and install it on the wall with few adjustments.\n option 4. C is an amateur woodworker. he was able to cut the wood to size and install it on the wall, but he took a long time to do so.;\n Correct answer: 3 ###\n",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="output", help="path to the output directory")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="path to the dataset directory")
    parser.add_argument('--question_file', type=str, default="egoschema_questionoptions.jsonl", help="name of the question file")
    parser.add_argument('--caption_file', type=str, default="blip2_egoschema_prompts.json", help="name of the validation file")
    parser.add_argument('--output_name', type=str, default="gpt4o_result.json", help="name of the output file")
    # gpt parameters
    parser.add_argument('--openai_key', type=str, required=True, help="openai key")
    parser.add_argument('--gpt_model', type=str, default='gpt-4o', help="GPT api name")
    parser.add_argument('--temperature', type=float, default=0.3, help="GPT api parameter temperature")
    parser.add_argument('--n', type=int, default=1, help="GPT api parameter n")
    parser.add_argument('--max_tokens', type=int, default=500, help="GPT api parameter max_tokens")
    args = parser.parse_args()
    
    output_path = os.path.join(args.output_dir, args.output_name)
    caption_path = os.path.join(args.dataset_dir, args.caption_file)
    question_path = os.path.join(args.dataset_dir, args.question_file)

    print('Caption data path: ', caption_path)
    print('Question data path: ', question_path)
    print('Result saving path: ', output_path)

    question_list = pd.read_json(question_path, lines=True)
    question_list = question_list['question'].tolist()
    cap_df = pd.read_json(caption_path, lines=False)
    cap_list = cap_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()

    client = OpenAI(api_key=args.openai_key)
    sample_idx = np.arange(len(cap_list)).tolist()
    total_num = len(sample_idx)
    over = False

    print("start ICL querying from {}".format(args.gpt_model))
    print('GPT api parameters: ', args.temperature, args.n, args.max_tokens)

    while not over:
        try:
            try:
                all_responses = json.load(open(output_path, "r"))
            except:
                all_responses = []
                json.dump(all_responses, open(output_path, "w"))
            print("processed sample num: ", len(all_responses)) 

            for ii, prompt_idx in enumerate(sample_idx):
                answers_list = []
                if ii < len(all_responses):
                    continue
                prompt_message = "After {{Correct answer:}}, you will predict the correct option NUMBER from the 5 options in the form of a SINGLE NUMBER with no text after. You ALWAYS have to output just a single NUMBER for what you think is the most correct option. \n"
                prompt_message += "Below is the video narrations.\n\n ### Narrations: " + f"{cap_list[prompt_idx]}" + "; " + f"{question_list[prompt_idx]}" "\n\n ### Correct answer: "
                mes = sys_message + [{'role': 'user', 'content': prompt_message}]

                start = time.perf_counter()

                response = client.chat.completions.create(model=args.gpt_model,
                                                          messages=mes,
                                                          max_tokens = args.max_tokens,
                                                          n = args.n,
                                                          temperature = args.temperature)
                answer = response.choices[0].message.content.strip()
                print("prediction: ", answer)
                number = answer
                counter = 0
                found = False
                try:
                    number = re.search(r'(\d+)', answer).group(1)
                except:
                    while found == False and counter < 5:
                        response = client.chat.completions.create(model=args.gpt_model,
                                                                  messages=mes,
                                                                  max_tokens = args.max_tokens,
                                                                  n = args.n,
                                                                  temperature = args.temperature)

                        print("try again")
                        answer = response.choices[0].message.content.strip()
                        print("prediction: ", answer)
                        counter += 1
                        try:
                            number = re.search(r'(\d+)', answer).group(1)
                            found = True
                            break
                        except:
                            print("didn't work")
                            continue 

                answers_list.append(number)

                e2e_inference_time = (time.perf_counter()-start)
                print(f"the inference time is {e2e_inference_time} s")
                print(str(ii+1)+'/'+str(total_num))
                all_responses.append(answers_list)
                json.dump(all_responses, open(output_path, "w"))    
            over = True
        except Exception as e:
            print(e)
