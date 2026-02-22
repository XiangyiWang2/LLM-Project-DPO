#!/bin/bash
export WANDB_PROJECT="qwen2.5-3b-dpo-alignment"

# 屏蔽坏卡/忙碌卡。比如你想用第1张卡（即物理卡1），就写1。它会被 PyTorch 当作 cuda:0
export CUDA_VISIBLE_DEVICES="1" 

echo "启动单卡 Debug/Baseline 训练..."
python src/dpo_train.py