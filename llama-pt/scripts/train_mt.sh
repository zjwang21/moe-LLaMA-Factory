source activate zhouh
export PATH=/usr/local/cuda-12.2/bin:$PATH
cd ../../
deepspeed --num_gpus 4 --master_port=9901 src/train_bash.py \
    --deepspeed /home/yangdezhao/zhouh_temp/LLaMA-Factory/llama-pt/config/ds_config_cpu.json \
    --stage sft \
    --model_name_or_path /home/yangdezhao/zhouh_temp/models/Llama-2-7b-hf \
    --flash_attn \
    --do_train \
    --task mt \
    --max_samples 1000000 \
    --template alpaca \
    --src_lang en \
    --tgt_lang is \
    --dataset opus100_enis \
    --preprocessing_num_workers 20 \
    --cutoff_len 512 \
    --finetuning_type moe \
    --moe_every_k_layers 1 \
    --moe_router_type top1 \
    --moe_num_experts 2 \
    --output_dir /home/yangdezhao/zhouh_temp/LLaMA-Factory/llama-pt/ckpts/moe-mt-enis-top1 \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --save_steps 500000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16
