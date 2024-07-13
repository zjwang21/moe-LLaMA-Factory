cd ../../
export WANDB_DISABLED=true
deepspeed --include="localhost:0,2,3,5,6" --master_port=9902 src/train_bash.py \
    --deepspeed /home/wangzj/LLaMA-Factory/llama-pt/config/ds_config_cpu.json \
    --stage pt \
    --flash_attn \
    --model_name_or_path /home/nfs02/wangzj/models/Qwen1.5-1.8B \
    --do_train \
    --dataset de_2b,ar_2b,hi_2b,is_2b \
    --preprocessing_num_workers 16 \
    --mix_strategy concat \
    --cutoff_len 1024 \
    --cache_path /home/nfs03/wangzj/dataset/pretrain/ardehiis8b \
    --finetuning_type ffn \
    --output_dir /home/nfs02/wangzj/checkpoints/llama-moe/qwen/1.8b-ardehiis8b-ffn \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16