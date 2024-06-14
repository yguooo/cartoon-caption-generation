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

## Finetuning

### SFT  
```
CUDA_VISIBLE_DEVICES=5 python humor_sft.py --output_dir /data/yguo/myoutput/  --dataset_dir /data/yguo/mydataset/
CUDA_VISIBLE_DEVICES=6 python humor_sft.py --output_dir /data/yguo/myoutput/  --dataset_dir /data/yguo/mydataset/ --new_padding_token
```

### DPO  
```
CUDA_VISIBLE_DEVICES=7 python humor_dpo.py --dataset_dir /data/yguo/mydataset/ --model_name mistralai/Mistral-7B-instruct-v0.1 --run_name full-instruct-dpo-warmup  --do_train --do_eval --output_dir /data/yguo/myoutput/
--model_checkpoint_name /data/yguo/myoutput/sft/new_pad/checkpoint-100
CUDA_VISIBLE_DEVICES=7 python humor_dpo.py --dataset_dir /data/yguo/mydataset/ --model_name mistralai/Mistral-7B-instruct-v0.1 --run_name full-instruct-dpo-warmup  --do_train --do_eval --output_dir /data/yguo/myoutput/
```

### Reward Modeling
```
CUDA_VISIBLE_DEVICES=7 python humor_reward_modeling.py --dataset_dir /data/yguo/mydataset/ --model_name weqweasdas/RM-Mistral-7B --run_name rm --do_train  --do_eval --output_dir /data/yguo/myoutput/ --max_steps 5000
CUDA_VISIBLE_DEVICES=7 python humor_reward_modeling.py --dataset_dir /data/yguo/mydataset/ --model_name weqweasdas/RM-Mistral-7B --run_name rm --do_train  --do_eval --output_dir /data/yguo/myoutput/ --max_steps 5000 --new_padding_token
```


# PPO
```
CUDA_VISIBLE_DEVICES=2 python humor_ppo.py --dataset_dir /data/yguo/mydataset --run_name ppo --output_dir /data/yguo/myoutput --target_kl 80 --reward_model mistralai/Mistral-7B-instruct-v0.1
```

# LLaVA finetune
```
deepspeed --include localhost:2 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --version v1 \
    --data_path /data/yguo/mydataset/llava_sft_dataset/train_llava_sft_dataset.json \
    --image_folder /data/yguo/mydataset/cartoons/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/yguo/myoutput/llava_sft/ \
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

## Refer to `finetuning/example_evaluation.py`

We provide all model checkpoints for finetuned models. Model checkpoints can be found [here](https://uwmadison.box.com/s/0c31rxhwgzqa5jvy7wd84qycjr1twf19).It incluces: 
- `reward`
- `sft`
- `dpo`
- `ppo`
- `llava_sft`

You can also see our sample caption generations from the pretrained model in this [folder](https://github.com/yguooo/cartoon-caption-generation/tree/main/finetuning/generation)

# Citation 

Please consider citing our work if you use our code and data in this repo 

```
Jain, L., Jamieson, K., Mankoff, R., Nowak, R., Sievert, S., (2020). The New Yorker Cartoon Caption Contest Dataset. https://nextml.github.io/caption-contest-data/
```



Our paper citation ? 