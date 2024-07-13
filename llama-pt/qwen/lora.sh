cd ../../
export WANDB_DISABLED=true
deepspeed --include="localhost:0,1" --master_port=9902 src/train_bash.py \
    --deepspeed /home/wangzj/LLaMA-Factory/llama-pt/config/ds_config.json \
    --stage pt \
    --flash_attn \
    --model_name_or_path /home/nfs02/wangzj/models/Qwen1.5-1.8B \
    --do_train \
    --dataset ar_2b,ru_2b,de_2b \
    --preprocessing_num_workers 16 \
    --mix_strategy concat \
    --cutoff_len 1024 \
    --cache_path /home/nfs03/wangzj/dataset/pretrain/arderu6b \
    --finetuning_type lora \
    --lora_target all \
    --output_dir /home/nfs02/wangzj/checkpoints/llama-moe/qwen/1.8b-arderu6b-lora \
    --overwrite_output_dir \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 10 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --save_steps 1000 \
    --learning_rate 2e-4 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16
