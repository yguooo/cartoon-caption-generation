# This script is modified from the example script in the following repository:
# https://github.com/huggingface/trl/ 
import os
import torch
import numpy as np
from datasets import load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
import wandb 
from datasets import load_from_disk
from dataclasses import dataclass
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from util import seed_everything
from transformers import Trainer, TrainingArguments, HfArgumentParser


seed_everything(2024)
tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """
    # data parameters
    dataset_dir: str = "/your/dataset/path/"
    # training parameters
    output_dir: str = "/your/output/directory/"
    # model parameters
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    # Choice of padding token does not affect the model performance, but they matter if we want to use the model as the
    # checkpoint for dpo/ppo models
    new_padding_token: bool = False
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        warmup_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1.41e-5, 
        report_to="wandb",
        optim="adamw_torch",
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        do_eval=True,
    )
    
    script_args.dataset_dir = os.path.join(script_args.dataset_dir, 'sft_dataset')  

    training_args.output_dir = os.path.join(training_args.output_dir, 'sft', script_args.model_name)  
    if script_args.new_padding_token:
        training_args.output_dir = os.path.join(training_args.output_dir, 'new_pad')
    
    WANDB_LOG_MODEL = True
    wandb.init(project="sft", name=os.path.join(script_args.model_name, training_args.output_dir))
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name) 
    train_sft = load_from_disk(os.path.join(script_args.dataset_dir, 'train_sft_dataset'))
    test_sft = load_from_disk(os.path.join(script_args.dataset_dir, 'test_sft_dataset')) 

    if script_args.new_padding_token:
        # Add new padding token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    else: 
        # Use eos token as padding token
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize_function(prompts, max_length = 512):
        '''
        Tokenize the prompt so that the model is only evaluated on the loss of the caption and insert an EOS token at 
        the end for proper termination.
        '''
        result = tokenizer([t + tokenizer.eos_token for t in prompts['text'] ], padding="max_length", max_length=max_length, truncation=True)
        result["labels"] = result["input_ids"].copy()
        result["labels"] = [[-100 if x == tokenizer.pad_token_id else x for x in lst] for lst in result["labels"]] 
        result["labels"] = [lst[:-1] + [tokenizer.eos_token_id] for lst in result["labels"]]  
        prompt_result = tokenizer([t for t in prompts['prompt'] ], truncation=True)
        tmp_labels = []
        for j, lst in enumerate(result["labels"]): 
            # find the total occurence of -100
            count = sum([i == -100 for i in lst])
            lst = list(lst)
            # Assign zero loss to the left padding + prompt part 
            mask_len = min(max_length, count+len(prompt_result[j]))
            lst[:mask_len] = [-100]*mask_len
            tmp_labels.append(lst) 
        result["labels"] = tmp_labels
        return result
    train_sft, test_sft = train_sft.map(tokenize_function, batched = True), test_sft.map(tokenize_function, batched = True)

    peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
    model = get_peft_model(model, peft_config)

    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_sft,
        eval_dataset=test_sft,
    )
    trainer.train()
    

