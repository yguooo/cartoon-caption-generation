# Save ZS result
CUDA_VISIBLE_DEVICES=5 python save_results.py \
    --method zs \
    --dataset_dir /data/yguo/mydataset \
    --output_dir /data/yguo/myoutput \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --num_generation 10

# Save SFT result
CUDA_VISIBLE_DEVICES=5 python save_results.py \
    --method sft --dataset_dir /data/yguo/mydataset \
    --output_dir /data/yguo/myoutput \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --model_checkpoint /data/yguo/myoutput/sft/new_pad/checkpoint-100 \
    --num_generation 10 \
    --new_padding_token 


    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Your dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Your dataset path")
    parser.add_argument("--method", type=str, required=True, help="Description for the caption generation method")
    parser.add_argument("--setting", type=str, default="", required=False, help="The setting name that you want to save the results as")
    parser.add_argument("--model_name", type=str, default=None, required="mistralai/Mistral-7B-Instruct-v0.1",\
        help="The pretrained model that your model is (finetuned from)")
    parser.add_argument("--model_checkpoint", type=str, default=None,required=False, help="Your model_checkpoint")
    parser.add_argument("--num_generation", type=int, default=10, required=False, help="Number of caption generations per contest")
    parser.add_argument("--new_padding_token", action="store_true", \
        help="Add a new padding token to the tokenizer, please keep it consistent with your model checkpoint")