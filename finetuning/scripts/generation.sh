# Save ZS result
CUDA_VISIBLE_DEVICES=7 python save_results.py \
    --method zs \
    --dataset_dir /your/dataset/dir \
    --output_dir /your/output/dir \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --num_generation 10

# Save bon result
CUDA_VISIBLE_DEVICES=0 python save_bon_results.py \
    --reward_model /your/finetuned/reward/model \
    --dataset_dir /your/dataset/dir \
    --generation_file /your/output/dir/generation/zs_gen10.csv \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --num_generation 10


# Save SFT result
CUDA_VISIBLE_DEVICES=5 python save_results.py \
    --method sft --dataset_dir /your/dataset/dir \
    --output_dir /your/output/dir \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --model_checkpoint /your/output/dir/sft/new_pad/checkpoint-100 \
    --num_generation 10 \
    --new_padding_token 

# Save dpo result
CUDA_VISIBLE_DEVICES=5 python save_results.py \
    --method dpo --dataset_dir /your/dataset/dir \
    --output_dir /your/output/dir \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --model_checkpoint /your/output/dir/sft/new_pad/checkpoint-100 \
    --num_generation 10 \
    --new_padding_token 

# Save ppo result
CUDA_VISIBLE_DEVICES=5 python save_results.py \
    --method ppo --dataset_dir /your/dataset/dir \
    --output_dir /your/output/dir \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --model_checkpoint mistralai/Mistral-7B-Instruct-v0.1 \
    --num_generation 10


# Save llava result
python save_results.py \
    --method llava --dataset_dir /your/dataset/dir \
    --output_dir /your/output/dir \
    --model_name llava-hf/llava-v1.6-mistral-7b-hf \
    --num_generation 10 \
    --device cuda:5

# Save llava_sft result
python save_results.py \
    --method llava --dataset_dir /your/dataset/dir \
    --output_dir /your/output/dir \
    --model_name llava-hf/llava-v1.6-mistral-7b-hf \
    --model_checkpoint your/llava/sft/checkpoint \
    --num_generation 10 \
    --device cuda:5
