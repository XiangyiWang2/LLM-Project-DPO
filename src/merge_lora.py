import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os 
def main():
    base_model_path = "Qwen/Qwen2.5-3B-Instruct"
    lora_path = "./qwen-3b-dpo-dropout/final_checkpoint"
    output_path = "./models/Qwen2.5-3B-DPO-Merged"
    print("Downloading the base model")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
    )

    print("Downloading the tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print("Downloading the LoRA")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging our Model")
    model=model.merge_and_unload()

    print("Saving the model")
    os.makedirs(output_path, exist_ok = True)
    model.save_pretrained(output_path, safe_serialization = True)

    tokenizer.save_pretrained(output_path)
if __name__ =="__main__":
    main()
