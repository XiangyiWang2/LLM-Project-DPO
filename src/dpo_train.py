import os
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
def load_config(config_path = "configs/dpo_configs.yaml"):
    with open(config_path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)
def main():
    config = load_config()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank!=-1

    if is_distributed:
        device_map = {"":local_rank}
        ds_config = "configs/ds_zero2.json"
    else:
        device_map = {"":0}
        ds_config = None
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
        device_map = device_map
    )

    peft_config = LoraConfig(
        r = config["lora_r"],
        lora_alpha = config["lora_alpha"],
        lora_dropout = config["lora_dropout"],
        target_modules = config["target_modules"],
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    dataset = load_from_disk(config["dataset_path"])
    dataset = dataset.train_test_split(test_size = 0.05, seed = 0)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    training_args = DPOConfig(
        output_dir=config["output_dir"],
        beta=config["beta"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        bf16=config["bf16"],
        report_to=config["report_to"], 
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
        remove_unused_columns=False, 
        gradient_checkpointing=True,
        deepspeed=ds_config,  
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(config["output_dir"], "final_checkpoint"))
if __name__ == "__main__":
    main()

    



