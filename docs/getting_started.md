## Contents
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## Dataset Preparation
### Stage 1: Pre-Training
For pre-training, we use the [LLaVA-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) to pretrain the mlp connector.

### Stage 2: Pre-FineTuning
For pre-finetuning, we use the [ALLaVA](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) caption data to warm-up the whole CuMo model.

### Stage 3: Visual Instruction Tuning
For the visual intruction tuning stage, we use the a mixture of datasets for training:

- [LLaVA-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
- [ShareGPT4V](https://sharegpt4v.github.io/)
- [LAION GPT4V](https://huggingface.co/datasets/laion/gpt4v-dataset)
- [DocVQA](https://www.docvqa.org/datasets/docvqa)
- [SynDog-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en)
- [ChartQA](https://github.com/vis-nlp/ChartQA)
- [DVQA](https://github.com/kushalkafle/DVQA_dataset)
- [AI2D](https://allenai.org/data/diagrams)
- [InfoVQA](https://www.docvqa.org/datasets/infographicvqa)
- [ALLaVA](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)
- [LIMA](https://huggingface.co/datasets/GAIR/lima)

Please download these datasets following the instructions and the [json files](https://huggingface.co/datasets/jiachenl/CuMo_dataset). The datasets are structured as:

```
CuMo
├── cumo
├── scripts
├── checkpoints
│   ├── CuMo-mistral-7b
│   ├── CuMo-mixtral-8x7b
├── data
│   ├── llava
│   │   ├── llava_pretrain
|   │   │   ├── images
|   │   │   ├── blip_laion_cc_sbu_558k.json              
│   ├── jsons 
│   │   ├── cumo_pft_allava.json
│   │   ├── cumo_vit_1649K.json
│   ├── coco
│   ├── gqa
│   ├── ocr_vqa
│   ├── textvqa
│   ├── share_textvqa
│   ├── vg
│   ├── gpt4v-dataset
│   ├── sam
│   ├── sharegpt4v
│   ├── wikiart
│   ├── web-celebrity
│   ├── web-landmark
│   ├── ALLaVA
│   ├── docvqa
│   ├── chartqa
│   ├── dvqa
│   ├── ai2d
│   ├── infovqa
│   ├── lima
│   ├── syndog-en
│   ├── eval
│   ├── ...
```
You can set $CuMo_DIR to specify the path to the root directory of the project.

## Training
After downloading the datasets and the JSON files, you can proceed to train the model using the following commands. Taking CuMo Mistral-7B as an example, the first step is to pre-train the MLP connector.
```bash
bash scripts/cumo/mistral_7b/pretrain_mistral_7b.sh
```

The next step is to pre-finetune the whole model,
```bash
bash scripts/cumo/mistral_7b/pft_mistral_7b.sh
```

The final step is the visual instruction tuning stage,
```bash
bash scripts/cumo/mistral_7b/sft_mistral_7b.sh
```

Note that these scripts are for training the model on a single node of 8xA100s. If you want to train the model on multiple nodes, you can use the [deepspeed](https://www.deepspeed.ai/getting-started/) multi-node trainings with added hostfile in the scripts.

## Evaluation
We evaluate CuMo models on multiple benchmarks and many scripts are based on [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) evaluation settings. We've adapted some of them into multi-GPU evaluation scripts and added evaluations on MMMU and Mathvista. You can download the checkpoints for CuMo [mistral-7b](https://huggingface.co/jiachenl/CuMo-mistral-7b) / [mixtral-8x7b](https://huggingface.co/jiachenl/CuMo-mixtral-8x7b) models and follow the evaluation instructions in [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to download the datasets accordingly. The datasets are structured as:

```
CuMo
├── cumo
├── scripts
├── checkpoints
│   ├── CuMo-mistral-7b
│   ├── CuMo-mixtral-8x7b
├── data
│   ├── eval
│   ├── ├── scienceqa
│   ├── ├── textvqa
│   ├── ├── pope
│   ├── ├── mme
│   ├── ├── gqa
│   ├── ├── seed
│   ├── ├── vqav2
│   ├── ├── mmvet
│   ├── ├── ...
```
Then run the following commands to evaluate the models. Here are examples based on CuMo Mistral-7b:

### ScienceQA
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/sqa_m.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### TextVQA
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/textvqa_m.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### POPE
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/pope_m.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### MME
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/mme_m.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### GQA
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/gqa.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### SEED
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/seed.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### VQAv2
Multi-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cumo/eval/vqav2.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### MM-Vet
Single-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0 sh scripts/cumo/eval/mmvet.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```
Then submit the cumo_mistral_7b.json to [MM-Vet Evluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).

### LLaVA-Wild
Single-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0 sh scripts/cumo/eval/llavabench.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```
Note that we use gpt-4-0613 for evaluation and you may specify your own API key for evaluation.

### MMBench
Single-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0 sh scripts/cumo/eval/mmbench.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```
Then submit the result to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).

### MMBench-CN
Single-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0 sh scripts/cumo/eval/mmbench_cn.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```
Then submit the result to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).

### MMMU
Single-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0 sh scripts/cumo/eval/mmmu.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```

### MathVista
Single-gpu inference 
```Shell
CUDA_VISIBLE_DEVICES=0 sh scripts/cumo/eval/mathvista.sh $CuMo_DIR/checkpoints/CuMo-mistral-7b mistralai/Mistral-7B-Instruct-v0.2
```
Note that we use gpt-3.5-turbo for evaluation and you may specify your own API key for evaluation.












