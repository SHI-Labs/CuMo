import os
import re
import time
import argparse
import json
from tqdm import tqdm

import sys
sys.path.append('../')
#from utilities import *

# OpenAI
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-01-25",
    api_key="input your own api key",
)

# load demo prompt
demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_chat_response_azure(promot, model="gpt-3.5-turbo", temperature=0, max_tokens=256, n=1, patience=10000000, sleep_time=0):
    #messages = [
    #    {"role": "user", "content": promot},
    #]
    # print("I am here")
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precis!ee assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': promot,
                }],
                temperature=temperature,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
                n=n
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "repetitive patterns" in str(e):
                promot = re.sub(r'(.+?)\1+', r'\1', promot)
                
            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(promot) * 0.9)
                new_start = len(promot) - new_size
                promot = promot[new_start:]
                messages = [
                    {"role": "user", "content": promot},
                ]
                
            if sleep_time > 0:
                time.sleep(5)
        time.sleep(1)
    return ""

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""
    
    if question_type == 'multi_choice' and response in choices:
        return response
    
    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = get_chat_response_azure(full_prompt)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--output_file', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='gpt-4-0613', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613'])
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    args = parser.parse_args()

    # args
    #import pdb
    #pdb.set_trace()
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != '':
        output_file = result_file.replace('.json', f'_{args.output_label}.json')
    else:
        output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    # full pids
    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print("Number of testing problems:", len(full_pids))

    # test pids
    if args.rerun:
        test_pids = full_pids
    else:
        test_pids = []
        for pid in full_pids:
            # print(pid)
            if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
                test_pids.append(pid)
    
    test_num = len(test_pids)
    print("Number of problems to run:", test_num)
    # print(test_pids)

    # tqdm, enumerate results
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]       

        
        extraction  = extract_answer(response, problem, args.quick_extract)
        results[pid]['extraction'] = extraction

        if i % args.save_every == 0 or i == test_num - 1:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
