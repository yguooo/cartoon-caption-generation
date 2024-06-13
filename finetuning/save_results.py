from datasets import load_from_disk
import pandas as pd
import random 
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoConfig
from peft import PeftModel
import argparse
from util import seed_everything

seed_everything(2024) 

def get_gen_config(MODEL_NAME):
    '''
    Our default generation configuration that performs top-p sampling.
    '''
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 256 # maximum number of new tokens that can be generated by the model
    generation_config.temperature = 0.7 # randomness of the generated tex
    generation_config.top_p = 0.95 # diversity of the generated text
    generation_config.do_sample = True # sampling during the generation process
    generation_config.repetition_penalty = 1.15 # the degree to which the model should avoid repeating tokens in the generated text
    return generation_config

def get_unique_dataset(df):
    '''
    Obtain the unique prompts from the dataset
    '''
    contest_numbers = []
    examples = []
    for example in df:
        if example['contest_number'] not in contest_numbers: 
            contest_numbers.append(example['contest_number'])
            examples.append({'contest_number': example['contest_number'], 'prompt': example['prompt']}) 
    return examples

def process_generation(cell):
    '''
    Only keep the caption and discard the explanation. For zero-shot model, explanation shows up even when not explicitly requested.
    '''
    # The generation text can be a non-string when the generation model fails.
    cell = str(cell) 
    cell = cell.strip()
    # Remove quotes
    while cell.startswith('"') and cell.endswith('"'):
        cell = (cell[1:-1]).strip()
    # Only keep the first line
    cell = cell.split('\n', 1)[0]
    # Remove quotes
    while cell.startswith('"') and cell.endswith('"'):
        cell = (cell[1:-1]).strip()
    return cell

################ Save ZS Result ################

def save_zs_results(MODEL_NAME, dataset_dir, output_dir, num_generation = 10):
    '''
    Save the caption generation result for the zero-shot model.
    '''
    test_dataset = load_from_disk(os.path.join(dataset_dir, 'zs_dataset', 'test_zs_dataset'))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
    )

    generation_config = get_gen_config(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    test_dataset_unique = get_unique_dataset(test_dataset)

    # For speed
    prompt_df = pd.DataFrame(test_dataset_unique)
    prompt_df = prompt_df.sort_values(by=['contest_number'], ascending=[True])
    for i in range(num_generation): 
        prompt_df["caption"+str(i+1)] = ""

    print(prompt_df)
    for i in range(len(prompt_df)): 
        row = prompt_df.iloc[i,:]
        
        texts = [row['prompt']]*num_generation
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(**encoding, generation_config=generation_config)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts = [gen[len(row['prompt'])+1 :] for gen in generated_texts]
        for j in range(num_generation): 
            # Only keep the caption and discard the explanation.
            prompt_df.iloc[i, j+2] = process_generation(generated_texts[j])
            
    prompt_df.to_csv(os.path.join(output_dir, 'zs_gen{}.csv'.format(num_generation)), index=False)
################ Save SFT Result ################

# from datasets import load_dataset, Dataset, load_from_disk
# import pandas as pd
# import random 
# import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="5"
# # Mistral and LangChain packages (prompt engineering)
# import torch
# from langchain import PromptTemplate, HuggingFacePipeline
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoConfig
# from peft import PeftModel

# for case in [200]:
#     MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
#     # dir = MODEL_NAME
#     # dir = "/data/yguo/output/sft-c13/ny-funny-caption-gpto-top1000/checkpoint-2000"
#     dir = "/data/yguo/output/sft-c17/ny-funny-caption-gpto4/checkpoint-{}".format(case)
#     # dir = "/data/yguo/output/sft-c12/checkpoint-1220" 
#     # dir = "/data/yguo/output/sft-c13/checkpoint-500" 
#     # dir = "/data/yguo/output/sft-c14/checkpoint-120" 
#     # dir = "/data/yguo/output/dpo/mistralai/Mistral-7B-instruct-v0.1single-instruct-dpo-mistral-warmup/checkpoint-110"


#     # train_sft = load_from_disk('data/train_sft_trl_gpto2_prompt_dataset') 
#     # test_sft = load_from_disk('data/test_sft_trl_gpto2_prompt_dataset') 
#     # validation_sft = load_from_disk('data/validation_sft_trl_gpto2_prompt_dataset')7 
 

#     train_sft = load_from_disk('/data/yguo/dataset/train_sft_trl_gpto4_prompt_dataset') 
#     test_sft = load_from_disk('/data/yguo/dataset/test_sft_trl_gpto4_prompt_dataset') 
#     validation_sft = load_from_disk('/data/yguo/dataset/validation_sft_trl_gpto4_prompt_dataset') 


#     def seed_everything(seed=2024):
#         # Python's built-in random module
#         random.seed(seed)
        
#         # Numpy
#         np.random.seed(seed)
        
#         # PyTorch
#         torch.manual_seed(seed)
        
#         # If using CUDA (GPU)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
#             # The following two lines ensure deterministic behavior but may impact performance:
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#     seed_everything(2024)

#     # quantization_config = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_compute_dtype=torch.float16,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_use_double_quant=True,
#     # )


#     # C13
#     # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     # model = AutoModelForCausalLM.from_pretrained(
#     #     MODEL_NAME, torch_dtype=torch.float16,
#     #     trust_remote_code=True,
#     #     device_map="cuda",
#     #     # quantization_config=quantization_config, 
#     # )
#     # peft_model = PeftModel.from_pretrained(
#     #     model=model,
#     #     model_id =dir,
#     #     device_map="cuda"
#     #     # device_map="auto",
#     # )
#     # model = peft_model

#     # C17

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME, torch_dtype=torch.float16,
#         trust_remote_code=True,
#         device_map="cuda",
#         # quantization_config=quantization_config, 
#     )
#     model.resize_token_embeddings(len(tokenizer))
#     peft_model = PeftModel.from_pretrained(
#         model=model,
#         model_id =dir,
#         device_map="cuda"
#         # device_map="auto",
#     )
#     model = peft_model

#     generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
#     generation_config.max_new_tokens = 256 # maximum number of new tokens that can be generated by the model
#     generation_config.temperature = 0.7 # randomness of the generated tex
#     generation_config.top_p = 0.95 #0.95 # diversity of the generated text
#     generation_config.do_sample = True # sampling during the generation process
#     generation_config.repetition_penalty = 1.15 # the degree to which the model should avoid repeating tokens in the generated text
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token

#     # pipe = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     return_full_text=True,
#     #     generation_config=generation_config,

#     # )
#     # llm = HuggingFacePipeline(pipeline=pipe)

#     # test_sft = load_from_disk('data/test_sft_trl_prompt_dataset') 
#     # train_sft = load_from_disk('data/train_sft_trl_prompt_dataset') 


#     def get_unique_dataset(df, info):
#         contest_numbers = []
#         examples = []
#         for example in df:
#             if example['contest_number'] not in contest_numbers: 
#                 contest_numbers.append(example['contest_number'])
#                 example['info'] = info
#                 examples.append(example) 
#         return examples
#     train_sft_unique = get_unique_dataset(train_sft, "train")
#     test_sft_unique = get_unique_dataset(test_sft, "test")
#     validation_sft_unique = get_unique_dataset(validation_sft, "validation")

#     num_generation = 50

#     # prompt_df = pd.DataFrame(train_sft_unique+test_sft_unique + validation_sft_unique)
#     prompt_df = pd.DataFrame(test_sft_unique + validation_sft_unique)
#     prompt_df = prompt_df[['contest_number', 'info', 'prompt']]
#     prompt_df = prompt_df.sort_values(by=['info', 'contest_number' ], ascending=[False, True])
#     for i in range(num_generation): 
#         prompt_df["caption"+str(i+1)] = ""

#     for i in range(len(prompt_df)): 
#         row = prompt_df.iloc[i,:]
        
#         texts = [row['prompt']]*num_generation
#         encoding = tokenizer(texts, padding=True, return_tensors='pt').to("cuda")
#         with torch.no_grad():
#             generated_ids = model.generate(**encoding, generation_config=generation_config)
#         generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         generated_texts = [gen[len(row['prompt'])+1 :] for gen in generated_texts]
#         for j in range(num_generation): 
#             prompt_df.iloc[i, j+3] = generated_texts[j]
#     prompt_df.to_csv('generation/sft_gpto4_iter{}_results_50gen.csv'.format(case), index=False)

# ################ Save DPO Result ################

# from datasets import load_dataset, Dataset, load_from_disk
# import pandas as pd
# import random 
# import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="7"
# # Mistral and LangChain packages (prompt engineering)
# import torch
# from langchain import PromptTemplate, HuggingFacePipeline
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoConfig
# from peft import PeftModel

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# for step in [400]:
#     dir = "/data/yguo/output/dpo/mistralai/mistral-7B-instruct-v0.1full-instruct-dpo-warmup/ny-funny-caption-gpto4/checkpoint-{}".format(step)
#     # dir = MODEL_NAME
#     # dir = "/data/yguo/output/dpo/mistralai/mistral-7B-instruct-v0.1full-instruct-dpo-warmup-no-warmup/ny-funny-caption-gpto/checkpoint-{}".format(case)
#     # dir = '/data/yguo/output/dpo/mistralai/mistral-7B-instruct-v0.1full-instruct-dpo-warmup/ny-funny-caption-gpto4/checkpoint-{}'.format(case)
#     #dir = "/data/yguo/output/dpo/retrain/mistralai/mistral-7B-instruct-v0.1full-instruct-dpo-warmup/ny-funny-caption-gpto/checkpoint-{}".format(step)
#     # dir = "/data/yguo/output/dpo/mistralai/mistral-7B-instruct-v0.1full-instruct-dpo-warmup/ny-funny-caption-gpto4/checkpoint-100"
#     # dir = "/data/yguo/output/dpo/mistralai/mistral-7B-instruct-v0.1full-instruct-dpo-warmup/ny-funny-caption-gpto-v1/checkpoint-100"
#     # dir = "/data/yguo/output/sft-c12/checkpoint-1220" 
#     # dir = "/data/yguo/output/sft-c13/checkpoint-500" 
#     # dir = "/data/yguo/output/sft-c14/checkpoint-120" 
#     # dir = "/data/yguo/output/dpo/mistralai/Mistral-7B-instruct-v0.1single-instruct-dpo-mistral-warmup/checkpoint-110"


#     def seed_everything(seed=2024):
#         # Python's built-in random module
#         random.seed(seed)
        
#         # Numpy
#         np.random.seed(seed)
        
#         # PyTorch
#         torch.manual_seed(seed)
        
#         # If using CUDA (GPU)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
#             # The following two lines ensure deterministic behavior but may impact performance:
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#     seed_everything(2024)

#     # quantization_config = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_compute_dtype=torch.float16,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_use_double_quant=True,
#     # )


#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME, torch_dtype=torch.float16,
#         trust_remote_code=True,
#         device_map="cuda",
#         # quantization_config=quantization_config, 
#     )
#     model.resize_token_embeddings(len(tokenizer))
#     peft_model = PeftModel.from_pretrained(
#         model=model,
#         model_id =dir,
#         device_map="cuda"
#         # device_map="auto",
#     )
#     model = peft_model

#     # model = AutoModelForCausalLM.from_pretrained(
#     #     dir, torch_dtype=torch.float16,
#     #     trust_remote_code=True,
#     #     device_map="auto",
#     #     # quantization_config=quantization_config, 
#     # )
#     # Configuration of some generation-related settings
#     generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
#     generation_config.max_new_tokens = 256 # maximum number of new tokens that can be generated by the model
#     generation_config.temperature = 0.7 # randomness of the generated tex
#     generation_config.top_p = 0.95 #0.95 # diversity of the generated text
#     generation_config.do_sample = True # sampling during the generation process
#     generation_config.repetition_penalty = 1.15 # the degree to which the model should avoid repeating tokens in the generated text
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token

#     # pipe = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     return_full_text=True,
#     #     generation_config=generation_config,

#     # )
#     # llm = HuggingFacePipeline(pipeline=pipe)

#     # test_sft = load_from_disk('data/test_sft_trl_prompt_dataset') 
#     # train_sft = load_from_disk('data/train_sft_trl_prompt_dataset') 

#     train_sft = load_from_disk('/data/yguo/dataset/train_sft_trl_gpto4_prompt_dataset') 
#     test_sft = load_from_disk('/data/yguo/dataset/test_sft_trl_gpto4_prompt_dataset') 
#     validation_sft = load_from_disk('/data/yguo/dataset/validation_sft_trl_gpto4_prompt_dataset') 

#     # train_sft = load_from_disk('/data/yguo/dataset/train_sft_trl_gpto_prompt_dataset') 
#     # test_sft = load_from_disk('/data/yguo/dataset/test_sft_trl_gpto_prompt_dataset') 
#     # validation_sft = load_from_disk('/data/yguo/dataset/validation_sft_trl_gpto_prompt_dataset') 

#     def get_unique_dataset(df, info):
#         contest_numbers = []
#         examples = []
#         for example in df:
#             if example['contest_number'] not in contest_numbers: 
#                 contest_numbers.append(example['contest_number'])
#                 example['info'] = info
#                 examples.append(example) 
#         return examples
#     train_sft_unique = get_unique_dataset(train_sft, "train")
#     test_sft_unique = get_unique_dataset(test_sft, "test")
#     validation_sft_unique = get_unique_dataset(validation_sft, "validation")

#     num_generation = 10

#     prompt_df = pd.DataFrame(train_sft_unique+test_sft_unique + validation_sft_unique)
#     prompt_df = prompt_df[['contest_number', 'info', 'prompt']]
#     prompt_df = prompt_df.sort_values(by=['info', 'contest_number' ], ascending=[False, True])
#     for i in range(num_generation): 
#         prompt_df["caption"+str(i+1)] = ""

#     for i in range(len(prompt_df)): 
#         row = prompt_df.iloc[i,:]
        
#         texts = [row['prompt']]*num_generation
#         encoding = tokenizer(texts, padding=True, return_tensors='pt').to("cuda")
#         with torch.no_grad():
#             generated_ids = model.generate(**encoding, generation_config=generation_config)
#         generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         generated_texts = [gen[len(row['prompt'])+1 :] for gen in generated_texts]
#         for j in range(num_generation): 
#             while generated_texts[j].startswith('"') and generated_texts[j].endswith('"'):
#                 generated_texts[j] = generated_texts[j][1:-1]
#             prompt_df.iloc[i, j+3] = generated_texts[j]
#     prompt_df.to_csv('dpo_gpto4_warmup_{}_results.csv'.format(step), index=False)

# ################ Save PPO Result ################
# from datasets import load_dataset, Dataset, load_from_disk
# import pandas as pd
# import random 
# import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="5"
# # Mistral and LangChain packages (prompt engineering)
# import torch
# from langchain import PromptTemplate, HuggingFacePipeline
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoConfig
# from peft import PeftModel

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# # dir = MODEL_NAME

# for iter_step in [0,1,2,5,7,10,20]: 
#     # dir = "/data/yguo/output/ppo/mistralai/Mistral-7B-Instruct-v0.1/ppo-rm-mistral-gpto/ny-funny-caption-gpto/checkpoint-{}".format(iter_step)
#     dir = "/data/yguo/output/ppo/nowarmup/lr3e-6/mistralai/Mistral-7B-Instruct-v0.1/ppo-rm-mistral-gpto-nowarmup/ny-funny-caption-gpto/checkpoint-{}".format(iter_step)
#     # dir = "/data/yguo/output/sft-c12/checkpoint-1220" 
#     # dir = "/data/yguo/output/sft-c13/checkpoint-500" 
#     # dir = "/data/yguo/output/sft-c14/checkpoint-120" 
#     # dir = "/data/yguo/output/dpo/mistralai/Mistral-7B-instruct-v0.1single-instruct-dpo-mistral-warmup/checkpoint-110"


#     def seed_everything(seed=2024):
#         # Python's built-in random module
#         random.seed(seed)
        
#         # Numpy
#         np.random.seed(seed)
        
#         # PyTorch
#         torch.manual_seed(seed)
        
#         # If using CUDA (GPU)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
#             # The following two lines ensure deterministic behavior but may impact performance:
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#     seed_everything(2024)

#     # quantization_config = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_compute_dtype=torch.float16,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_use_double_quant=True,
#     # )


#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     # model = AutoModelForCausalLM.from_pretrained(
#     #     MODEL_NAME, torch_dtype=torch.float16,
#     #     trust_remote_code=True,
#     #     device_map="cuda",
#     #     # quantization_config=quantization_config, 
#     # )

#     # peft_model = PeftModel.from_pretrained(
#     #     model=model,
#     #     model_id =dir,
#     #     device_map="cuda"
#     #     # device_map="auto",
#     # )
#     # model = peft_model
    
    
#     model = AutoModelForCausalLM.from_pretrained(
#         dir, torch_dtype=torch.float16,
#         trust_remote_code=True,
#         device_map="cuda",
#         # quantization_config=quantization_config, 
#     )
    
    
#     # Configuration of some generation-related settings
#     generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
#     generation_config.max_new_tokens = 256 # maximum number of new tokens that can be generated by the model
#     generation_config.temperature = 0.7 # randomness of the generated tex
#     generation_config.top_p = 0.95 #0.95 # diversity of the generated text
#     generation_config.do_sample = True # sampling during the generation process
#     generation_config.repetition_penalty = 1.15 # the degree to which the model should avoid repeating tokens in the generated text
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token

#     # pipe = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     return_full_text=True,
#     #     generation_config=generation_config,

#     # )
#     # llm = HuggingFacePipeline(pipeline=pipe)

#     # test_sft = load_from_disk('data/test_sft_trl_prompt_dataset') 
#     # train_sft = load_from_disk('data/train_sft_trl_prompt_dataset') 

#     train_sft = load_from_disk('data/train_sft_trl_gpto_prompt_dataset') 
#     test_sft = load_from_disk('data/test_sft_trl_gpto_prompt_dataset') 
#     validation_sft = load_from_disk('data/validation_sft_trl_gpto_prompt_dataset') 

#     def get_unique_dataset(df, info):
#         contest_numbers = []
#         examples = []
#         for example in df:
#             if example['contest_number'] not in contest_numbers: 
#                 contest_numbers.append(example['contest_number'])
#                 example['info'] = info
#                 examples.append(example) 
#         return examples
#     train_sft_unique = get_unique_dataset(train_sft, "train")
#     test_sft_unique = get_unique_dataset(test_sft, "test")
#     validation_sft_unique = get_unique_dataset(validation_sft, "validation")

#     num_generation = 10

#     # prompt_df = pd.DataFrame(train_sft_unique+test_sft_unique + validation_sft_unique)
#     prompt_df = pd.DataFrame(test_sft_unique + validation_sft_unique)
#     prompt_df = prompt_df[['contest_number', 'info', 'prompt']]
#     prompt_df = prompt_df.sort_values(by=['info', 'contest_number' ], ascending=[False, True])
#     for i in range(num_generation): 
#         prompt_df["caption"+str(i+1)] = ""

#     for i in range(len(prompt_df)): 
#         row = prompt_df.iloc[i,:]
        
#         texts = [row['prompt']]*num_generation
#         encoding = tokenizer(texts, padding=True, return_tensors='pt').to("cuda")
#         with torch.no_grad():
#             generated_ids = model.generate(**encoding, generation_config=generation_config)
#         generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         generated_texts = [gen[len(row['prompt'])+1 :] for gen in generated_texts]
#         for j in range(num_generation): 
#             while generated_texts[j].startswith('"') and generated_texts[j].endswith('"'):
#                 generated_texts[j] = generated_texts[j][1:-1]
#             prompt_df.iloc[i, j+3] = generated_texts[j]
#     prompt_df.to_csv('ppo_nowarmup_lr3e-6_step{}_results.csv'.format(iter_step), index=False)
# ################ Save llava Result ################
# import torch
# from transformers import pipeline, BitsAndBytesConfig
# import pandas as pd
# import pickle

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import os
# device = "cuda:0"

# import torch
# from tqdm import tqdm
# tqdm.pandas()
# import pandas as pd
# import pickle

# import pandas as pd
# import os
# # Mistral and LangChain packages (prompt engineering)
# import torch
# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoConfig

# import json

# # gpto descriptions
# with open('gpto_descriptions.pickle', 'rb') as file:
#     description_data = pickle.load(file)
# train_examples = description_data['train_samples']
# test_examples = description_data['test_samples']
# validation_examples = description_data['validation_samples']


# # def create_llava_inputs(examples, info):
# #     dt = []
# #     for example in examples:
# #         if example['contest_number'] == 525: # Ranking data for this contest is not available
# #             continue
# #         entities = example['entities']
# #         prompt =   """image: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image:"""\
# #             .format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
# #         dt.append({
# #             "contest_number": example['contest_number'],
# #             "info": info,
# #             "image": str(example['contest_number'])+".jpg",
# #             "prompt": prompt
# #         })
# #     return dt



# # Case 2 more complex prompt

# # def create_llava_inputs(examples, info):
# #     dt = []
# #     for example in examples:
# #         if example['contest_number'] == 525: # Ranking data for this contest is not available
# #             continue
# #         entities = example['entities']
# #         prompt =   """I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will provide a cartoon image with descriptions and then you should generate 1 funny caption for the cartoon along with an explanation for each. \nimage: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image:"""\
# #             .format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
# #         dt.append({
# #             "contest_number": example['contest_number'],
# #             "info": info,
# #             "image": str(example['contest_number'])+".jpg",
# #             "prompt": prompt
# #         })
# #     return dt

# # Case 3 more complex prompt with separator

# def create_llava_inputs(examples, info):
#     dt = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         entities = example['entities']
#         prompt =   """[INST]  I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will provide a cartoon image with descriptions and then you should generate 1 funny caption for the cartoon along with an explanation for each. \nimage: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image: [/INST]"""\
#             .format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         dt.append({
#             "contest_number": example['contest_number'],
#             "info": info,
#             "image": str(example['contest_number'])+".jpg",
#             "prompt": prompt
#         })
#     return dt


# train = create_llava_inputs(train_examples, "train")
# test = create_llava_inputs(test_examples, "test")
# validation = create_llava_inputs(validation_examples, "validation")


# num_generation = 50

# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# processor = LlavaNextProcessor.from_pretrained(model_id)

# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)


# # prompt_df = pd.DataFrame(train+test+validation)
# prompt_df = pd.DataFrame(test+validation) # Onlyl keep the test and validation for time efficiency
# prompt_df = prompt_df[['contest_number', 'info', 'image' , 'prompt']]
# prompt_df = prompt_df.sort_values(by=['info', 'contest_number' ], ascending=[False, True])
# for i in range(num_generation): 
#     prompt_df["caption"+str(i+1)] = ""


# from transformers import GenerationConfig
# generation_config = GenerationConfig.from_pretrained(model_id)

# generation_config.temperature = 0.7 # randomness of the generated tex
# generation_config.top_p = 0.95 # diversity of the generated text
# generation_config.do_sample = True # sampling during the generation process
# generation_config.repetition_penalty = 1.15 # the degree to which the model should avoid repeating tokens in the generated text

# for i in range(len(prompt_df)): 
#     row = prompt_df.iloc[i,:]
#     image = Image.open("/data/yguo/hf-dataset/newyorker_caption_ranking/cartoons/"+row['image'])
#     prompt = row['prompt']
#     for j in range(num_generation): 
#         inputs = processor(prompt, image, return_tensors="pt").to(device)
#         # autoregressively complete prompt
#         output = model.generate(**inputs, max_new_tokens=256, generation_config=generation_config)
#         output = processor.decode(output[0], skip_special_tokens=True)
#         output = output[len(prompt):]
#         prompt_df.iloc[i, j+4] = output # Only keep the caption and discard the explanation.

# prompt_df.to_csv('llava_zs_gpto3_results_50gen.csv', index=False)
# ################ Save llava SFT Result ################

# import torch
# from transformers import pipeline, BitsAndBytesConfig
# import pandas as pd
# import pickle

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import os
# device = "cuda:4"

# import torch
# from tqdm import tqdm
# tqdm.pandas()
# import pandas as pd
# import pickle

# import pandas as pd
# import os
# # Mistral and LangChain packages (prompt engineering)
# import torch
# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoConfig

# import json

# # gpto descriptions
# with open('gpto_descriptions.pickle', 'rb') as file:
#     description_data = pickle.load(file)
# train_examples = description_data['train_samples']
# test_examples = description_data['test_samples']
# validation_examples = description_data['validation_samples']


# def create_llava_inputs(examples, info):
#     dt = []
#     for example in examples:
#         if example['contest_number'] == 525: # Ranking data for this contest is not available
#             continue
#         entities = example['entities']
#         # prompt =   """image: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image:"""\
#         prompt = """[INST]  I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will provide a cartoon image with descriptions and then you should generate 1 funny caption for the cartoon along with an explanation for each. \nimage: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image: [/INST]""".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
#         dt.append({
#             "contest_number": example['contest_number'],
#             "info": info,
#             "image": str(example['contest_number'])+".jpg",
#             "prompt": prompt
#         })
#     return dt

# train = create_llava_inputs(train_examples, "train")
# test = create_llava_inputs(test_examples, "test")
# validation = create_llava_inputs(validation_examples, "validation")


# num_generation = 50

# for step in [11000]:
#     model_id = "/data/yguo/output/llava-sft/c3/checkpoint-{}".format(step)
#     processor = LlavaNextProcessor.from_pretrained(model_id)

#     model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)


#     prompt_df = pd.DataFrame(test+validation) # Use only test + validation for efficiency
#     prompt_df = prompt_df[['contest_number', 'info', 'image' , 'prompt']]
#     prompt_df = prompt_df.sort_values(by=['info', 'contest_number' ], ascending=[False, True])
#     for i in range(num_generation): 
#         prompt_df["caption"+str(i+1)] = ""


#     from transformers import GenerationConfig
#     generation_config = GenerationConfig.from_pretrained(model_id)

#     generation_config.temperature = 0.7 # randomness of the generated tex
#     generation_config.top_p = 0.95 # diversity of the generated text
#     generation_config.do_sample = True # sampling during the generation process
#     generation_config.repetition_penalty = 1.15 # the degree to which the model should avoid repeating tokens in the generated text

#     for i in range(len(prompt_df)): 
#         row = prompt_df.iloc[i,:]
#         image = Image.open("/data/yguo/hf-dataset/newyorker_caption_ranking/cartoons/"+row['image'])
#         prompt = row['prompt']
#         for j in range(num_generation): 
#             inputs = processor(prompt, image, return_tensors="pt").to(device)
#             # autoregressively complete prompt
#             output = model.generate(**inputs, max_new_tokens=256, generation_config=generation_config)
#             output = processor.decode(output[0], skip_special_tokens=True)
#             output = output[len(prompt):]
#             prompt_df.iloc[i, j+4] = output # Only keep the caption and discard the explanation.

#     prompt_df.to_csv('llava_sft_gpto_c3_{}_results_50gen.csv'.format(step), index=False)

def main(args):
    args.output_dir = os.path.join(args.output_dir, 'generation')
    if args.method == "zs":
        save_zs_results(args.model_name, args.dataset_dir, args.output_dir, num_generation = 10)
    else: 
        raise ValueError("Pick a valid method from [zs, sft, dpo, ppo, llava, llava_sft] ")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Your dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Your dataset path")
    parser.add_argument("--method", type=str, required=True, help="Description for the caption generation method")
    parser.add_argument("--model_name", type=str, default=None, required="mistralai/Mistral-7B-Instruct-v0.1",\
        help="The pretrained model that your model is (finetuned from)")
    parser.add_argument("--model_checkpoint", type=str, default=None,required=False, help="Your model_checkpoint")
    parser.add_argument("--num_generation", type=int, default=10, required=False, help="Number of caption generations per contest")

    args = parser.parse_args()
    main(args)
