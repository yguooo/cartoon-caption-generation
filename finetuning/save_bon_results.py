import os
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import argparse
from util import seed_everything
from datasets import load_from_disk
from save_results import get_unique_dataset


def pick_bon(row, reward_df, sentiment_pipe, n = 10): 
    '''
    Pick the top n captions given a row of the dataframe
    '''
    contest_number = row['contest_number']
    print(contest_number)
    captions = row[2:].to_list() 
    prompt = reward_df[reward_df['contest_number'] == contest_number]['prompt'].tolist()[0]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Your dataset path")
    parser.add_argument("--reward_model", type=str, required=True, help="Your reward model directory")
    parser.add_argument("--generation_file", type=str, required=True, help="filename of the caption generation")
    parser.add_argument("--model_name", type=str, default=None, required="mistralai/Mistral-7B-Instruct-v0.1",\
        help="The pretrained model that your model is (finetuned from)")
    parser.add_argument("--num_generation", type=int, default=10, required=False, help="Number of caption generations per contest")
    
    args = parser.parse_args()
    seed_everything(2024) 
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        device_map="cuda",
        num_labels=1,
    )
    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model,
        tokenizer=tokenizer,
    )

    df = pd.read_csv(args.generation_file)
    df_topn = df.copy()[:args.num_generation+2]
    # Note that we will need to load the reward dataset since the reward model is trained on the reward dataset with 
    # possibly simpler prompts than those finetuned models
    reward_test_dataset = load_from_disk(os.path.join(args.dataset_dir, 'dpo_dataset', 'test_dpo_dataset')) 
    reward_test_dataset_unique = get_unique_dataset(reward_test_dataset)
    reward_df = pd.DataFrame(reward_test_dataset_unique)
    
    for i in range(len(df)): 
        row = df.iloc[i,:]
        bon_texts = pick_bon(row, reward_df, sentiment_pipe, n = 10)
        for j in range(len(bon_texts)): 
            df_topn.iloc[i, j+2] = bon_texts[j] # Only keep the caption and discard the explanation.

    df_topn.to_csv(args.generation_file[:-4] + "_BoN.csv", index=False)
