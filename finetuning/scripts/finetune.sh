
# SFT  

CUDA_VISIBLE_DEVICES=5 python humor_sft.py --output_dir /your/output/dir/  --dataset_dir /your/dataset/dir/
CUDA_VISIBLE_DEVICES=6 python humor_sft.py --output_dir /your/output/dir/  --dataset_dir /your/dataset/dir/ --new_padding_token


# DPO  
CUDA_VISIBLE_DEVICES=7 python humor_dpo.py --dataset_dir /your/dataset/dir/ --model_name mistralai/Mistral-7B-instruct-v0.1 --run_name full-instruct-dpo-warmup  --do_train --do_eval --output_dir /your/output/dir/
--model_checkpoint_name /your/sft/checkpoint/with/simple/prompt/

# Reward Modeling
CUDA_VISIBLE_DEVICES=2 python humor_ppo.py --dataset_dir /your/dataset/dir --run_name ppo --output_dir /your/output/dir --target_kl 80 --reward_model /your/finetuned/reward/model

# PPO
CUDA_VISIBLE_DEVICES=2 python humor_ppo.py --dataset_dir /your/dataset/dir/ --run_name ppo --output_dir /your/output/dir/ --target_kl 80 --reward_model /your/finetuned/reward/model

# LLaVA finetune

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