import os
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

def load_config(config_path="configs/dpo_config.yaml"):
    """读取 YAML 配置文件，做到代码与配置解耦"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    print(f"[*] Starting DPO Training for {cfg['model_name_or_path']}")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1 
    print(f"[*] Distributed Environment Detected: {is_distributed}")

    if is_distributed:
        device_map = {"": local_rank} 
        ds_config = "configs/ds_zero2.json"
    else:
        device_map = {"": 0}         
        ds_config = None

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[*] Loading model with FlashAttention-2...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        device_map=device_map
    )
    print("[*] Configuring LoRA...")
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"[*] Loading dataset from {cfg['dataset_path']}...")
    dataset = load_from_disk(cfg["dataset_path"])
    
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    training_args = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=cfg["beta"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy=cfg["evaluation_strategy"],
        eval_steps=cfg["eval_steps"],
        bf16=cfg["bf16"],
        report_to=cfg["report_to"], 
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
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

    print("[*] Launching DPO Training...")
    trainer.train()
    print(f"[*] Training complete. Saving LoRA adapters to {cfg['output_dir']}/final_checkpoint")
    trainer.save_model(os.path.join(cfg["output_dir"], "final_checkpoint"))

if __name__ == "__main__":
    main()