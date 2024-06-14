# Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning 

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



## Dataset
Read our paper at arxiv (link)

See past hosted contest at this [website](https://nextml.github.io/caption-contest-data/)

This code constains code for: 
- Finetuning preference models
- Evaluation with languages models like GPT-4 
- results exploration and diversity investigation

## Dataset Statistics

## Evaluation
We present a novel multimodal preference dataset for creative tasks, consisting of over 250 million human ratings on more than 2.2 million captions, collected through crowdsourcing rating data for The New Yorker's weekly cartoon caption contest over the past eight years. This unique dataset supports the development and evaluation of multimodal large language models and preference-based fine-tuning algorithms for humorous caption generation. We propose novel benchmarks for judging the quality of model-generated captions, utilizing both GPT4 and human judgments to establish ranking-based evaluation strategies. Our experimental results highlight the limitations of current fine-tuning methods, such as RLHF and DPO, when applied to creative tasks. Furthermore, we demonstrate that even state-of-the-art models like GPT4 and Claude currently underperform top human contestants in generating humorous captions. As we conclude this extensive data collection effort, we release the entire preference dataset to the research community, fostering further advancements in AI humor generation and evaluation.

## Finetuning
Prior to finetuning, you need to change the directory.
```
cd finetuning
```
and create several datasets with the following command:
```
python preprocess.py 
```

### SFT  
```
python humor_sft.py --output_dir /your/output/dir/  --dataset_dir /your/dataset/dir/
```

### DPO  
Our experiments show that using a sft checkpoint from simple prompt forms the better checkpoint for DPO than training from scratch or from an SFT checkpoint with long prompt.
```
python humor_sft.py --output_dir /your/output/dir/  --dataset_dir /your/dataset/dir/ --new_padding_token --simple_prompt
```

Including `--new_padding_token` will produce similar model, but it is required to obtain an sft checkpoint for further finetuning the DPO model.  
Including `--simple_prompt` will use a simple prompt (same as DPO) for SFT. 

Then, you can finetune the DPO from this generated SFT checkpoint.
```
python humor_dpo.py --dataset_dir /your/dataset/dir/ --model_name mistralai/Mistral-7B-instruct-v0.1 --run_name full-instruct-dpo-warmup  --do_train --do_eval --output_dir /your/output/dir/
--model_checkpoint_name /your/sft/checkpoint/with/simple/prompt/
```

### Reward Modeling
You can use the following command to finetune a reward model. Since generating humorous texts typically is not in the training dataset of public reward model, we need to finetune the reward model ourselves.
```
python humor_reward_modeling.py --dataset_dir /your/dataset/dir/ --model_name weqweasdas/RM-Mistral-7B --run_name rm --do_train  --do_eval --output_dir /your/output/dir/ --max_steps 5000
```
You can also choose custom reward model from [reward bench](https://huggingface.co/spaces/allenai/reward-bench) to finetune different reward models.


### PPO
Our PPO model is directly finetune from `mistralai/Mistral-7B-instruct-v0.1`. You need to first finetune a reward model to run the PPO.
```
python humor_ppo.py --dataset_dir /your/dataset/dir --run_name ppo --output_dir /your/output/dir --target_kl 80 --reward_model /your/finetuned/reward/model
```

### LLaVA finetune
To perform LLaVA finetune, you need to first clone the original LLaVA directory. 
```
git clone https://github.com/haotian-liu/LLaVA/
```
Then, at the uppermost level, run the following command.
```
deepspeed --include localhost:2 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --version v1 \
    --data_path /your/dataset/dir/llava_sft_dataset/train_llava_sft_dataset.json \
    --image_folder /your/dataset/dir/cartoons/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /your/output/dir/llava_sft/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

## Generating the results of pretrained and finetuned language model 

You can generate sample captions given an trained model using the following commands. 
```
# Save ZS result
python save_results.py --method zs --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name mistralai/Mistral-7B-Instruct-v0.1 --num_generation 10
# Save SFT result
CUDA_VISIBLE_DEVICES=5 python save_results.py --method sft --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_checkpoint /your/output/dir/sft/new_pad --num_generation 10 --new_padding_token 
# Save dpo result
CUDA_VISIBLE_DEVICES=5 python save_results.py --method dpo --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_checkpoint /your/output/dir/sft/new_pad --num_generation 10 --new_padding_token 
# Save ppo result
python save_results.py --method ppo --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_checkpoint mistralai/Mistral-7B-Instruct-v0.1 --num_generation 10
# Save llava result
python save_results.py --method llava --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name llava-hf/llava-v1.6-mistral-7b-hf --num_generation 10 --device cuda:5
# Save llava sft result
python save_results.py --method llava --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name llava-hf/llava-v1.6-mistral-7b-hf --model_checkpoint your/llava/sft/checkpoint --num_generation 10 --device cuda:5
```
To obtain the best-of-N sample generation, you need to first generate more captions. We recommend generating 5 times more captions than the final generations as a rule of thumb. Then, you can pick good captions out of these generations with a finetuned reward model.
```
python save_results.py --method zs --dataset_dir /your/dataset/dir --output_dir /your/output/dir --model_name mistralai/Mistral-7B-Instruct-v0.1 --num_generation 50
python save_bon_results.py --reward_model /your/reward/model/ --dataset_dir /your/dataset/dir --generation_file /your/output/dir/generation/zs_gen10.csv --model_name mistralai/Mistral-7B-Instruct-v0.1 --num_generation 10
```

You can also check out our already generated captions in `examples/generation`. To further evaluate these generations, you can also refer to `finetuning/generation_evaluation.ipynb` or `ranking/example_rank_more.ipynb`. 

## Download Checkpoints
Since the finetune procedure can take from 1 day up to a week on an A100, we provide all model checkpoints for finetuned models. Model checkpoints can be found [here](https://uwmadison.box.com/s/0c31rxhwgzqa5jvy7wd84qycjr1twf19).It incluces: 
- `reward`
- `sft`
- `dpo`
- `ppo`
- `llava_sft`

You can also see our sample caption generations from the pretrained model in this `examples/generations`
- `claude.csv`: 10 captions generated from Claude-3-Opus
- `gpt4o.csv`: 10 captions generated from GPT-4o Vision
- `zs.csv`: 10 captions generated from Mistral-Instruct-7B in a zero shot manner 
- `zs_BoN.csv`: We first generated 50 captions using Mistral-Instruct-7B in a zero-shot manner, then we use our finetuned reward model to pick the best 10 captions.
- `sft.csv`: 10 captions generated from sft model of Mistral-Instruct-7B
- `dpo.csv`: 10 captions generated from DPO finetuned model of Mistral-Instruct-7B
- `ppo.csv`: 10 captions generated from PPO finetuned model of Mistral-Instruct-7B
- `llava.csv`: 10 captions generated from LLaVA pretrained model (llava-v1.6-mistral-7b-hf)
- `llava_sft.csv`: 10 captions generated from LLaVA finetuned model from (llava-v1.6-mistral-7b-hf)


# Citation 

Please consider citing our work if you use our code and data in this repo 

```
Jain, L., Jamieson, K., Mankoff, R., Nowak, R., Sievert, S., (2020). The New Yorker Cartoon Caption Contest Dataset. https://nextml.github.io/caption-contest-data/
```



Our paper citation ? 
