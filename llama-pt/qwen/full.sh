cd ../../
export WANDB_DISABLED=true
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port=9902 src/train_bash.py \
    --deepspeed /root/huangxin/nanda/llm-moe-merge/config/ds.json \
    --stage pt \
    --flash_attn \
    --model_name_or_path /root/huangxin/nanda/model/Qwen1.5-0.5B \
    --do_train \
    --dataset bn \
    --preprocessing_num_workers 128 \
    --cache_dir /mnt/vol1/huangxin/nanda/dataset/CulturaX/cache \
    --cache_path /mnt/vol1/huangxin/nanda/dataset/CulturaX/qwen-tokenized-bn \
    --cutoff_len 2048 \
    --finetuning_type full \
    --output_dir /root/huangxin/nanda/checkpoints/merge/0.5b-bn \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16