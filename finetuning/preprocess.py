from datasets import load_dataset, Dataset
from huggingface_hub import login, Repository
import pandas as pd
import random
import os
from util import seed_everything
import numpy as np
import json

def preprocess_zs(examples, sample_pairs_per_prompt=1000): 
    '''
    Process the prompt for zero-shot evaluation 
    '''
    prompts = []
    for example in examples:
        ranking_data = pd.read_csv(os.path.join(DATA_PATH, "ranking", str(example['contest_number'])+'.csv'))
        entities = example['entities']
        prompt =   "[INST] <> I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon. \n scene: {} \n description: {} \n uncanny description: {} \n entities: {} <> \n funny caption: [/INST]".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
        idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
        for i in idx: 
            prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
    return prompts

def preprocess_sft(examples, sample_pairs_per_prompt = 1000):
    '''
    Process the prompt for supervised finetuning 
    '''
    prompts = []
    for example in examples:
        ranking_data = pd.read_csv(os.path.join(DATA_PATH, "ranking", str(example['contest_number'])+'.csv'))
        entities = example['entities']
        prompt =  "[INST]I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters.  Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct.  It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 1 funny caption for the cartoon[/INST] \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nfunny caption:".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
        idx = random.sample(range(len(ranking_data)), sample_pairs_per_prompt) 
        for i in idx: 
            prompts.append({"text": prompt+str(ranking_data.iloc[i]['caption']), "prompt": prompt, 'contest_number': example['contest_number']})
    return prompts

def preprocess_dpo(examples, sample_pairs_per_prompt=1000):
    '''
    Process the prompt for DPO/Reward modeling/PPO 
    '''
    dpo_data = {
        'contest_number': [],
        'prompt': [], 
        'chosen': [],
        'rejected': []
    }
    i = 0 
    for example in examples:
        ranking_data = pd.read_csv(os.path.join(DATA_PATH, "ranking", str(example['contest_number'])+'.csv'))
        sampled_pairs = 0
        counter = 0
        while sampled_pairs < sample_pairs_per_prompt and counter < 200000 : 
            chosen_caption_id = random.randint(0, int(0.5*len(ranking_data)))
            rejected_caption_id = random.randint(chosen_caption_id +1, len(ranking_data)-1)
            counter += 1
            if  ranking_data.iloc[chosen_caption_id]['mean'] - ranking_data.iloc[rejected_caption_id]['mean']  > \
       3*np.sqrt((ranking_data.iloc[chosen_caption_id]['precision']**2 + ranking_data.iloc[rejected_caption_id]['precision']**2)): 
                entities = example['entities']
                prompt = "scene: {} \n description: {} \n uncanny description: {} \n entities: {} \n funny caption: ".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
                dpo_data['contest_number'].append(example['contest_number'])
                dpo_data['prompt'].append(prompt)
                dpo_data['chosen'].append(str(ranking_data.iloc[chosen_caption_id]['caption']))
                dpo_data['rejected'].append(str(ranking_data.iloc[rejected_caption_id]['caption']))
                sampled_pairs += 1
    return Dataset.from_dict(dpo_data)


def preprocess_llava(examples, info):
    '''
    Process the prompt for LLAVA/LLaVA SFT in the evaluation setting
    
    In our original paper, we evaluate the reward model only on the Hessel test split instead of the combined split of 
    test and validation. Since each contest contains 1000 comparisons, it is sufficient for the evaluation.
    '''
    llava_data = []
    for example in examples:
        entities = example['entities']
        prompt = """[INST]  I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will provide a cartoon image with descriptions and then you should generate 1 funny caption for the cartoon along with an explanation for each. \nimage: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image: [/INST]""".format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
        llava_data.append({
            "contest_number": example['contest_number'],
            "info": info,
            "image": str(example['contest_number'])+".jpg",
            "prompt": prompt
        })
    return llava_data

def process_llava_sft(examples, k = 1000):
    '''
    Process the prompt for LLaVA in the finetuning setting
    
    Note that we need to modify the format of the prompt for the LLaVA finetuning procedure as it only the json format. 
    The specific propmting procedure still remains the same. 
    '''
    prompts = []
    for example in examples:
        ranking_data = pd.read_csv(os.path.join(DATA_PATH, "ranking", str(example['contest_number'])+'.csv'))
        entities = example['entities']
        prompt =   """[INST]  I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions. \n First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters. Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct. It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will provide a cartoon image with descriptions and then you should generate 1 funny caption for the cartoon along with an explanation for each. \nimage: <image> \nscene: {} \ndescription: {} \nuncanny description: {} \nentities: {} \nGenerate a funny caption for the image: [/INST]"""\
            .format(example['location'],example['canny'], example['uncanny'], ', '.join(entities)) 
        idx = range(min(len(ranking_data), k))
        for i in idx: 
            prompts.append({
                "id": example['contest_number'], 
                "image": str(example['contest_number'])+".jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt
                    },
                    {
                        "from": "gpt",
                        "value": str(ranking_data.iloc[i]['caption'])
                    },
                ]})    
    return prompts

if __name__ == "__main__":
    DATA_PATH = "/your/data/path/"
    login(token="your_huggingface_token")
    
    df = load_dataset('yguooo/newyorker_caption_ranking', "gpt4o_description") 

    repo = Repository(local_dir=DATA_PATH, clone_from="https://huggingface.co/datasets/yguooo/newyorker_caption_ranking")

    df_train = df['train'].to_pandas()
    # We keep the data before contest 890 since the contest after 890 was not available before when we performed the experiment.
    df_train = df_train[df_train['contest_number'] < 890].to_dict(orient='records')
    
    # Our finetuning experiment combine the test split and validation split of Hessel's, since our evaluation is
    # per-contest basis. We need to the ensure the test split is large enough.
    df_test = df['test'].to_pandas() 
    df_validation = df['validation'].to_pandas() 
    df_test = pd.concat([df_test,df_validation], axis=0).to_dict(orient='records')

    seed_everything(2024)
    
    # Save the zs prompt 
    train_zs_data, test_zs_data  = preprocess_zs(df_train), preprocess_zs(df_test)
    train_zs_data, test_zs_data = Dataset.from_list(train_zs_data), Dataset.from_list(test_zs_data)

    train_zs_data.save_to_disk(os.path.join(DATA_PATH, 'zs_dataset', 'train_zs_dataset'))
    test_zs_data.save_to_disk(os.path.join(DATA_PATH, 'zs_dataset', 'test_zs_dataset'))
    
    # Save the SFT prompt 
    train_sft_data, test_sft_data  = preprocess_sft(df_train), preprocess_sft(df_test)
    train_sft_data, test_sft_data = Dataset.from_list(train_sft_data), Dataset.from_list(test_sft_data)

    train_sft_data.save_to_disk(os.path.join(DATA_PATH, 'sft_dataset' , 'train_sft_dataset'))
    test_sft_data.save_to_disk(os.path.join(DATA_PATH, 'sft_dataset' , 'test_sft_dataset'))

    # Save the DPO prompt 
    
    train_dpo_data = preprocess_dpo(df_train)
    test_dpo_data = preprocess_dpo(df_test) 

    train_dpo_data.save_to_disk(os.path.join(DATA_PATH, 'dpo_dataset', 'train_dpo_dataset'))
    test_dpo_data.save_to_disk(os.path.join(DATA_PATH, 'dpo_dataset', 'test_dpo_dataset')) 
  
    # Save the LLaVA prompt for evaluation purposes
    
    train_llava_data = preprocess_llava(df_train, "train")
    test_llava_data = preprocess_llava(df_test, "test")
    train_llava_data = Dataset.from_list(train_llava_data)
    test_llava_data = Dataset.from_list(test_llava_data)

    train_llava_data.save_to_disk(os.path.join(DATA_PATH, 'llava_dataset', 'train_llava_dataset'))
    test_llava_data.save_to_disk(os.path.join(DATA_PATH, 'llava_dataset', 'test_llava_dataset')) 
    
    # Save the LLaVA prompt for finetuning purposes
    
    train_llava_sft_data, test_llava_sft_data = process_llava_sft(df_train), process_llava_sft(df_test)

    if not os.path.exists(os.path.join(DATA_PATH, 'llava_sft_dataset')):
        os.makedirs(os.path.join(DATA_PATH, 'llava_sft_dataset'))

    with open(os.path.join(DATA_PATH, 'llava_sft_dataset', 'train_llava_sft_dataset.json'), 'w') as file:
        file.write(json.dumps(train_llava_sft_data))
    with open(os.path.join(DATA_PATH, 'llava_sft_dataset', 'test_llava_sft_dataset.json'), 'w') as file:
        file.write(json.dumps(test_llava_sft_data))
