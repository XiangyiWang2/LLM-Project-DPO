#!/bin/bash
export WANDB_PROJECT="qwen2.5-3b-dpo-alignment"


export CUDA_VISIBLE_DEVICES="1,2,3,4,6,7" 

echo "启动多卡 DeepSpeed 分布式训练..."

torchrun --nproc_per_node=6 \
    --master_port=29500 \
    src/dpo_train.py