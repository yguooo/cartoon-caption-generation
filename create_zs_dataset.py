from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
#import tyro
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardConfig, RewardTrainer
tqdm.pandas()
import pandas as pd
import random
from peft import LoraConfig
import pickle

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpto descriptions
with open('gpto_descriptions.pickle', 'rb') as file:
    description_data = pickle.load(file)
train_examples = description_data['train_samples']
test_examples = description_data['test_samples']
validation_examples = description_data['validation_samples']


# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "[INST] <> I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon along with an explanation for each. \n scene: {} \n description: {} \n uncanny description: {} \n entities: {} <> \n funny caption: [\INST]".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('data/train_zs_prompt_dataset')
# test_sft_trl.save_to_disk('data/test_zs_prompt_dataset')
# validation_sft_trl.save_to_disk('data/validation_zs_prompt_dataset')


# On removing the requirement for explanation

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "[INST] <> I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \n scene: {} \n description: {} \n uncanny description: {} \n entities: {} <> \n funny caption: [\INST]".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs2_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs2_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs2_prompt_dataset')

# On alternative format for the prompt

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "<s>[INST]I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {}[/INST]\nfunny caption:</s>".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs3_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs3_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs3_prompt_dataset')

# Simpler Prompt: no instruction for how to generate funny caption

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "<s>[INST] You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {}[/INST]\nfunny caption:</s>".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs4_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs4_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs4_prompt_dataset')

# # Remove the mistral formatting

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nfunny caption:".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs5_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs5_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs5_prompt_dataset')

# # Simpler format: Only ask the question at the end

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "scene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \n From the cartoon descriptions above, generate a funny caption for the cartoon:".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs6_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs6_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs6_prompt_dataset')


# # Simplest format: Naive prompt

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "scene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nfunny caption:".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs7_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs7_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs7_prompt_dataset')


# Remove the <> 

# def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
#     prompts = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
#         entities = example['entities']
#         prompt =   "[INST] I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \n scene: {} \n description: {} \n uncanny description: {} \n entities: {} \n funny caption: [\INST]".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         if sample_pairs_per_prompt == None:
#             idx = range(len(ranking_data))
#         else: 
#             idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
#         for i in idx: 
#             prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
#     return prompts
# train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
# train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

# train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs8_prompt_dataset')
# test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs8_prompt_dataset')
# validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs8_prompt_dataset')

# rearrange [\INST]

def create_sft_dataset_gpt(examples, sample_pairs_per_prompt = None):
    prompts = []
    for example in examples:
        if example['contest_number'] == 525: # Ranking data for this contest is not available
            continue
        ranking_data = pd.read_csv('data/'+str(example['contest_number'])+'.csv')
        entities = example['entities']
        prompt =   "[INST] I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \n scene: {} \n description: {} \n uncanny description: {} \n entities: {} [\INST] \n funny caption: ".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
        if sample_pairs_per_prompt == None:
            idx = range(len(ranking_data))
        else: 
            idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
        for i in idx: 
            prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
    return prompts
train_sft_data, test_sft_data, validation_sft_data  = create_sft_dataset_gpt(train_examples, 1000), create_sft_dataset_gpt(test_examples, 1000), create_sft_dataset_gpt(validation_examples, 1000)
train_sft_trl, test_sft_trl, validation_sft_trl = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data), Dataset.from_list(validation_sft_data) 

train_sft_trl.save_to_disk('/data/yguo/dataset/train_zs9_prompt_dataset')
test_sft_trl.save_to_disk('/data/yguo/dataset/test_zs9_prompt_dataset')
validation_sft_trl.save_to_disk('/data/yguo/dataset/validation_zs9_prompt_dataset')


