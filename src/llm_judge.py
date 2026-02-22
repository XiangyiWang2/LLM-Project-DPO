
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os
import matplotlib.pyplot as plt
TEST_PROMPTS = [
   
    "What specific statistical tests can be applied to analyze the difference in height of tomato plants between group A (2L water) and group B (1L water)? Provide multi-step reasoning, consider sample size/variance, and use LaTeX for explanations of null hypothesis, p-value, and effect size.",
    

    "Write a song about 'loss' in pop/ballad style. Requirements: 1. Use poetic imagery. 2. Output lyrics in Markdown. 3. Use 'dream mode' for creative lines denoted with **. 4. Include key/chords in the lyrics. 5. Do not offer to produce a demo.",
    

    "Translate this sentence to Hindi without changing the meaning: 'Just a few days ago, Ambassadors of approximately eighty-five countries went to Amritsar from Delhi.' Provide the solution directly.",
    

    "Write a 'Cat in the Hat' style rhyme for a 2nd grader about being healthy. You MUST use EVERY one of these words: cat, hat, sat, rat, mat, pat, chat, flat, splat, man, fan, can, pan, van, plan, clan, span, scan.",
    

    "Explain the advantages of automatic instrumentation with OpenTelemetry. Focus on developer experience and scalability."
]


def generate_responses(model_path, prompts):
    """加载模型，生成回答，然后彻底卸载释放显存"""
    print(f"\n[🔄] 正在加载选手模型: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    responses = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response_text)
        print(f"  - 完成生成 {i+1}/{len(prompts)}")
        
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return responses

def judge_responses(judge_model_path, prompts, base_resps, dpo_resps):
    """加载裁判模型，进行盲测打分"""
    print(f"\n[⚖️] 正在加载裁判模型 (Judge): {judge_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    results = {"DPO_Win": 0, "Base_Win": 0, "Tie": 0}
    judge_template =   """You are a strict quality control auditor for Large Language Models.
Your goal is to determine which assistant followed the user's instructions more PRECISELY. Be less likely to give a tie please.

[User Prompt]
{prompt}

[Assistant A]
{answer_a}

[Assistant B]
{answer_b}

[Evaluation Task]
1. Check Constraints: Did the assistant include ALL requested elements (e.g., specific words, LaTeX, Markdown formatting, specific chords)?
2. Logical Depth: Is the reasoning sound or just surface-level?
3. Formatting: Is the output structured as requested?

A fails if it misses even one minor constraint. B wins if it is more comprehensive and follows all 'Additional Instructions'.

Conclusion: A wins / B wins / Tie
"""

    for i in range(len(prompts)):
        prompt = judge_template.format(
            prompt=prompts[i], 
            answer_a=base_resps[i], 
            answer_b=dpo_resps[i]
        )
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        judgment = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\n题目 {i+1} 裁判结果: {judgment}")
        
        # 匹配英文裁判结果
        if "B wins" in judgment:
            results["DPO_Win"] += 1
        elif "A wins" in judgment:
            results["Base_Win"] += 1
        else:
            results["Tie"] += 1

    return results

def main():
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    dpo_model = "./models/Qwen2.5-3B-DPO-Merged"
    judge_model = "Qwen/Qwen2.5-7B-Instruct" 

    base_responses = generate_responses(base_model, TEST_PROMPTS)
    dpo_responses = generate_responses(dpo_model, TEST_PROMPTS)
    results = judge_responses(judge_model, TEST_PROMPTS, base_responses, dpo_responses)
    
    total = len(TEST_PROMPTS)
    print("\n" + "="*40)
    print("🏆 LLM-as-a-Judge 盲测最终结果 (English In-Domain) 🏆")
    print(f"评测集大小: {total} 条指令")
    print(f"基座模型 (Base) 胜率: {results['Base_Win']/total*100:.1f}%")
    print(f"DPO模型 (Merged) 胜率: {results['DPO_Win']/total*100:.1f}%")
    print(f"平局率 (Tie): {results['Tie']/total*100:.1f}%")
    print("="*40)

    # 画图部分
    labels = ['Base Win', 'Tie', 'DPO Win']
    counts = [results['Base_Win'], results['Tie'], results['DPO_Win']]
    colors = ['#7b9ce6', '#d3d3d3', '#f4a261'] 
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=colors, width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title('LLM-as-a-Judge: English In-Domain Evaluation', fontsize=14, pad=15)
    plt.ylabel('Number of Wins', fontsize=12)
    plt.ylim(0, max(counts) + 1 if max(counts) > 0 else 5) 
    
    output_dir = "./qwen2.5-3b-dpo-output"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "english_win_rate_chart.png")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 英文评测可视化图表已自动生成并保存至: {plot_path}")

if __name__ == "__main__":
    main()