export WANDB_PROJECT="qwen2.5-3b-dpo-alignment"
export CUDA_VISIBLE_DEVICES="1" 
python src/dpo_train.py
