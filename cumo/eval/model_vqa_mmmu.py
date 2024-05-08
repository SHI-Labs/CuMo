import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from cumo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cumo.conversation import conv_templates, SeparatorStyle
from cumo.model.builder import load_pretrained_model
from cumo.utils import disable_torch_init
from cumo.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from datasets import load_dataset, concatenate_datasets

from cumo.eval.mmmu_utils.data_utils import load_yaml, save_json, CAT_SHORT2LONG
from cumo.eval.mmmu_utils.eval_utils import parse_multi_choice_response, parse_open_response

from PIL import Image
import math
import re

def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = re.findall("<img='(.*?)'>", option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': data['image_1'], 'question_type': data['question_type']}

def construct_prompt(sample):
    question = sample['question']
    options = eval(sample['options'])
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt = question + '\n' + example + '\n' + "Answer with the option's letter from the given choices directly"
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        empty_prompt = question + '\n' + "Answer the question using a single word or phrase."
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.config.training = False

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        print("loading ", subject)
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    out_samples = dict()

    for sample in tqdm(dataset, total=len(dataset)):
        sample = process_single_sample(sample)

        sample = construct_prompt(sample)
        
        qs = sample['final_input_prompt'].replace('<image 1>', '').strip()

        if sample['image'] is not None:
            image_tensor = process_images([sample['image'].convert('RGB')], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [sample['image'].size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            images = None
            image_sizes = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                #temperature=args.temperature,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if sample['question_type'] == 'multiple-choice':
            pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
        else:  # open question
            pred_ans = response
        out_samples[sample['id']] = pred_ans
    
    save_json(answers_file, out_samples)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
