# This script is modified from the example script in the following repository:
# https://github.com/huggingface/trl/ 
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_from_disk, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, pipeline, HfArgumentParser
import os
from trl import RewardConfig, is_xpu_available
import wandb
from custom_trainer import CustomRewardTrainer
from util import seed_everything
seed_everything(2024)


@dataclass
class ScriptArguments:
    run_name: str = "reward_modeling"
    model_name: str = "weqweasdas/RM-Mistral-7B"
    """the model name"""
    dataset_dir: str = "/your/dataset/path"
    """dataset name"""
    dataset_text_field: str = "text"
    """the text field of the dataset"""
    eval_split: str = "none"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    load_in_8bit: bool = False
    """load the model in 8 bits precision"""
    do_train: bool = False
    """Do training"""
    do_eval: bool = False
    """New Padding Token"""
    new_padding_token: bool = False
    """Do evaluation"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    output_dir: str = "/your/output/path/"
    """output directory"""
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir = "", # placeholder for output directory
            per_device_train_batch_size=4,
            num_train_epochs=1,
            # max_steps=1000,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            report_to=["wandb"],
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=400,
            save_steps=400,
            eval_steps=400, 
            save_total_limit=20,
            max_length=512,
            bf16=True,
            save_strategy="steps",
            evaluation_strategy="steps",
        )
    )
    max_steps: int = 1000
    checkpoint: str = None
    """Checkpoint"""
    resume: bool = False
    """Whether to resume"""
    use_peft: bool = True
    eval_dataset_name: str = None
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head",],
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        ),
    )

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.reward_config.output_dir = args.output_dir
    args.reward_config.max_steps = args.max_steps
    args.reward_config.output_dir = os.path.join(args.reward_config.output_dir, 'reward_modeling', args.model_name, args.run_name)
    if args.new_padding_token:
        args.reward_config.output_dir = os.path.join(args.reward_config.output_dir, "new_pad")


    WANDB_LOG_MODEL = True
    wandb.init(project= "reward_model", name=args.reward_config.output_dir)
    tqdm.pandas()

    # Step 1: Load the model
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
    else:
        device_map = None
        quantization_config = None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        num_labels=1,
    )

    # Step 2: Load the dataset and pre-process it
    train_dataset = load_from_disk(os.path.join(args.dataset_dir, 'dpo_dataset', 'train_dpo_dataset'))
    eval_dataset = load_from_disk(os.path.join(args.dataset_dir, 'dpo_dataset', 'test_dpo_dataset'))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.new_padding_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    def preprocess_function(pairs, max_length = 512): 
        new_examples = {
            'input_ids_chosen': [],
            'attention_mask_chosen': [],
            'input_ids_rejected': [],
            'attention_mask_rejected': [],
        }
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            rejected_encodings_dict = tokenizer(
                rejected,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            new_examples['input_ids_chosen'].append(chosen_encodings_dict["input_ids"])
            new_examples['attention_mask_chosen'].append(chosen_encodings_dict["attention_mask"])
            new_examples['input_ids_rejected'].append(rejected_encodings_dict["input_ids"])
            new_examples['attention_mask_rejected'].append(rejected_encodings_dict["attention_mask"])
        return new_examples

    train_dataset = Dataset.from_dict(preprocess_function(train_dataset))
    eval_dataset = Dataset.from_dict(preprocess_function(eval_dataset))

    # Step 4: Define the LoraConfig
    if args.use_peft:
        peft_config = args.peft_config
    else:
        peft_config = None

    def compute_metrics(eval_pred):
        logits, label = eval_pred 
        acc = (logits[:,0] - logits[:,1] >= 0).mean()
        return {"acc": acc}


    # Step 5: Define the Trainer
    trainer = CustomRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args.reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=compute_metrics
    )
    if args.do_train: 
        if args.resume: 
            trainer.load(checkpoint = args.checkpoint)
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    if args.do_eval: 
        trainer.load(checkpoint = args.checkpoint)
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")



