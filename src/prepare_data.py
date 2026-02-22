import os
from datasets import load_dataset
from transformers import AutoTokenizer

def main():

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[*] Loading Tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    print("[*] Downloading UltraFeedback dataset (subset)...")
    raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:20000]")

    def format_dpo_data(example):
        """
        核心逻辑：将原始数据转化为 DPOTrainer 识别的 prompt, chosen, rejected 格式。
        必须严格使用 Qwen2.5 的 apply_chat_template 来构建上下文！
        """
        messages = [
            {"role": "system", "content": "You are a helpful, respectful, and honest assistant."},
            {"role": "user", "content": example["prompt"]}
        ]
        
        # 使用 Qwen 的专属模板引擎，拼接出 <|im_start|>user\n... 这样的标准格式
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        chosen_content = example["chosen"][-1]["content"]
        rejected_content = example["rejected"][-1]["content"]
        chosen_str = chosen_content + tokenizer.eos_token
        rejected_str = rejected_content + tokenizer.eos_token

        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str
        }

    print("[*] Formatting dataset to Qwen ChatML format...")
    dpo_dataset = raw_dataset.map(format_dpo_data, num_proc=8, remove_columns=raw_dataset.column_names)
    output_dir = "./data/qwen2.5_dpo_dataset"
    os.makedirs(output_dir, exist_ok=True)
    dpo_dataset.save_to_disk(output_dir)
    print(f"[*] Successfully saved {len(dpo_dataset)} formatted DPO pairs to {output_dir}")

if __name__ == "__main__":
    main()