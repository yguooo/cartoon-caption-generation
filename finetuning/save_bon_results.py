from dataclasses import dataclass, field
import os
from typing import Optional
from test_code import test_code
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, HfArgumentParser, AutoModelForCausalLM, AutoModelForSequenceClassification, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed, PreTrainedModelWrapper
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
import wandb
import pandas as pd
import pickle
import numpy as np

dir_list = ["generation/zs_gpto11_results_50gen_parsed.csv"
    # "generation/BoN/zs_gpto_results_50gen_parsed.csv", # Best ZS
    # "generation/BoN/dpo_gpto_400_parsed_results_50gen_parsed.csv", # Best DPO
    #"generation/BoN/llava_zs_gpto3_results_50gen_parsed_new.csv", # Best LLAVA zs
    # "generation/BoN/ppo_top1000_30_results50gen_parsed.csv", # Best PPO
    # "generation/BoN/sft_gpto4_iter200_results_50gen_parsed.csv", # Best SFT
    # "generation/BoN/llava_sft_gpto_c3_11000_results_50gen_parsed.csv", # Best LLAVA SFT
    # "generation/BoN/zs_gpto10_results_parsed_50gen.csv"  
]


# Load the reward model

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1") # args.ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

reward_model="/data/yguo/output/humor_reward_modeling/weqweasdas/RM-Mistral-7Bfull-eos/ny-funny-caption-gpto/checkpoint-3400"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model,
    # quantization_config=quantization_config,
    device_map="cuda:5",
    num_labels=1,
)
reward_model.config.pad_token_id = reward_model.config.eos_token_id
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    tokenizer=tokenizer,
    # device="cuda"
)


for dir in dir_list: 
    n = 10

    df = pd.read_csv(dir)

    dir = dir[:-4] # Remove extension

    df_topn = df.iloc[:,:(n+3)]

    # gpto descriptions
    with open('gpto_descriptions.pickle', 'rb') as file:
        description_data = pickle.load(file)
    train_examples = description_data['train_samples']
    test_examples = description_data['test_samples']
    validation_examples = description_data['validation_samples']

    examples = { i['contest_number']:i for i in (train_examples + validation_examples + test_examples)}

    # Find the top captions given a row

    def pick_bon(row, n = 10): 
        contest_number = row['contest_number']
        print(contest_number)
        captions = row[3:53].to_list() # 50 captions per contest
        example = examples[contest_number]
        prompt =  "scene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nfunny caption:".format(example['location'],example['canny'], example['uncanny'], ', '.join(example['entities'])) 
        texts = [ prompt + str(caption) for caption in captions]
        sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = np.array([torch.tensor(output[0]["score"]) for output in pipe_outputs])
        
        top_captions = []
        orders = np.argsort(rewards)
        for (i,order) in enumerate(orders): 
            if order in np.arange(len(orders)-n, len(orders)): 
                top_captions.append(captions[i])
        return top_captions


    for i in range(len(df)): 
        row = df.iloc[i,:]
        bon_texts = pick_bon(row, n = 10)
        for j in range(n): 
            df_topn.iloc[i, j+3] = bon_texts[j] # Only keep the caption and discard the explanation.

    df_topn.to_csv(dir + "_BoN.csv", index=False)




################ fix CSV


