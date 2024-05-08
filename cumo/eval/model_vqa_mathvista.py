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

from PIL import Image
import math
import re

def process_single_sample(data):
    return {'id': data['id'], 'question': data['question'], 'options': data['options'], 'answer': data['answer'], 'image': data['decoded_image'], 'question_type': data['question_type']}

def construct_prompt(sample):
    question = sample['question']
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
        #empty_prompt_sample_structure = config['multi_choice_example_format']
        #empty_prompt = empty_prompt_sample_structure.format(question, example)
        empty_prompt = question + '\n' + example + '\n' + "Answer with the option's letter from the given choices directly"
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['empty_prompt'] = empty_prompt
        res_dict['final_input_prompt'] = empty_prompt
    elif sample['question_type'] == 'free_form':
        empty_prompt = question + '\n' + "Answer the question using a single word or phrase."
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        res_dict['final_input_prompt'] = empty_prompt

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
    dataset = load_dataset(args.data_path, split=args.split)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    out_samples = dict()
    for ind, sample in enumerate(tqdm(dataset, total=len(dataset))):
        pid = sample['pid']
        
        qs = sample['question']

        if sample['decoded_image'] is not None:
            #image_file = line["image"]
            #image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([sample['decoded_image'].convert('RGB')], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [sample['decoded_image'].size]
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
        sample['response'] = response
        del sample['decoded_image']
        out_samples[pid] = sample
    
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
    parser.add_argument('--data_path', type=str, default="AI4Math/MathVista") # hf dataset path.
    parser.add_argument('--split', type=str, default='testmini')
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
