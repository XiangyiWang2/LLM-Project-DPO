import os
from datasets import load_dataset
from transformers import AutoTokenizer
def main():
    model = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    raw_data = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split = "train_prefs[:20000]")
    def format(example):
        messages = [
            {"role": "system", "content":"You are a helpful, respectful and honest assistant."},
            {"role": "user", "content": example["prompt"]}    
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        chosen_content = example["chosen"][-1]["content"]
        rejected_content = example["rejected"][-1]["content"]
        chosen_str = chosen_content+tokenizer.eos_token
        rejected_str = rejected_content+tokenizer.eos_token
        return {"prompt":prompt_str, "chosen" : chosen_str, "rejected":rejected_str}
    dpo_dataset = raw_data.map(format, num_proc = 8, remove_columns = raw_data.column_names)
    output_dir = "./data/qwen2.5_dpo_dataset"
    os.makedirs(output_dir, exist_ok = True)
    dpo_dataset.save_to_disk(output_dir)
if __name__ == "__main__":
    main()
