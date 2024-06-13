# This script is modified from the example script in the following repository:
# https://github.com/huggingface/trl/ 
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import PartialState
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
import wandb
from trl import is_xpu_available
from custom_trainer import CustomDPOTrainer
from util import seed_everything
import os 
seed_everything(2024)



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    # training parameters
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    model_checkpoint_name: Optional[str] = field(default="", metadata={"help": "the model checkpoint name"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "optimizer learning rate"})  
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the number of gradient accumulation steps"}
    )
    output_dir: Optional[str] = field(default="/your/output/directory/", metadata={"help": "the output directory"})
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Whether to activate fp16 mixed precision during training"}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "Whether to activate bf16 mixed precision during training"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    #num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of training epochs"}),  
    max_steps: Optional[int] = field(default=1500, metadata={"help": "max number of training steps"}) # 2200 steps
    # lora parameters
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    checkpoint: str = None
    resume: bool = False
    do_train: bool = False
    do_eval: bool = False
    run_name: str = ""
    dataset_dir: str = "/your/dataset/path/"
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    generate_during_eval: Optional[bool] = field(default=False, metadata={"help": "Generate during evaluation"})

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args.output_dir = os.path.join(script_args.output_dir, 'dpo', script_args.model_name, script_args.run_name) 

    WANDB_LOG_MODEL = True
    wandb.init(project="dpo", name=script_args.output_dir + script_args.model_name)

    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{PartialState().local_process_index}"}
            if is_xpu_available()
            else {"": PartialState().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        

    if script_args.model_checkpoint_name == "": 
        # Case 1: dpo from a pretrained checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        model.resize_token_embeddings(len(tokenizer)) 
    else:
        # Case 2: dpo from a sft checkpoint with padded token.
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        ) 
        model.resize_token_embeddings(len(tokenizer)) 
        model = PeftModel.from_pretrained(
            model=model,
            model_id=script_args.model_checkpoint_name,
            device_map="auto",
        )

    if not script_args.use_peft:
        model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name)
    else:
        # If one uses PEFT, there is no need to load a reference model
        model_ref = None


    train_dataset = load_from_disk(os.path.join(script_args.dataset_dir, 'dpo_dataset', 'train_dpo_dataset'))
    eval_dataset = load_from_disk(os.path.join(script_args.dataset_dir, 'dpo_dataset', 'test_dpo_dataset'))
    
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        # num_train_epochs=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=100,  # match results in blog post
        eval_steps=200,
        save_steps=200,
        output_dir=script_args.output_dir,
        optim="rmsprop",
        warmup_steps=150, 
        report_to=script_args.report_to,
        bf16=script_args.bf16,
        save_total_limit=50,
        fp16=script_args.fp16,
        gradient_checkpointing=script_args.gradient_checkpointing,
    )

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    print("model")
    print(model)
    print(model.num_parameters())
    
    
    # 5. initialize the DPO trainer
    dpo_trainer = CustomDPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=script_args.generate_during_eval,
        peft_config=peft_config,
    )

    # 6. train

    if script_args.do_train: 
        if script_args.resume: 
            dpo_trainer.load(checkpoint = script_args.checkpoint)
        train_result = dpo_trainer.train()
        metrics = train_result.metrics
        dpo_trainer.log_metrics("train", metrics)
        dpo_trainer.save_metrics("train", metrics)
    if script_args.do_eval: 
        dpo_trainer.load(checkpoint = script_args.checkpoint)
        metrics = dpo_trainer.evaluate(eval_dataset=eval_dataset)
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)
        dpo_trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")

