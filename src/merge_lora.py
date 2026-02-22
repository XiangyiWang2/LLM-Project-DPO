# src/merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    # 1. 路径设置
    # 注意：如果你的 base_model 是下载在本地 models 文件夹的，这里替换成本地路径
    base_model_path = "Qwen/Qwen2.5-3B-Instruct" 
    lora_path = "./qwen2.5-3b-dpo-output/final_checkpoint"
    output_path = "./models/Qwen2.5-3B-DPO-Merged"

    print(f"[*] 正在加载基座模型 (Base Model): {base_model_path} ...")
    # 加载原版模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", # 让 transformers 自动分配显存
    )
    
    print("[*] 正在加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"[*] 正在加载 LoRA 权重: {lora_path} ...")
    # 把 LoRA 挂载到基座模型上
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("[*] 开始执行融合 (merge_and_unload) ... 这可能需要一两分钟。")
    # 核心操作：将 LoRA 矩阵乘法合并进原本的 Linear 层
    model = model.merge_and_unload()

    print(f"[*] 融合完成！正在保存完整模型到: {output_path} ...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True) # 保存为 safetensors 格式
    tokenizer.save_pretrained(output_path)
    
    print("[*] 所有步骤完成！你现在拥有了一个完全独立的 DPO 模型！🎉")

if __name__ == "__main__":
    main()