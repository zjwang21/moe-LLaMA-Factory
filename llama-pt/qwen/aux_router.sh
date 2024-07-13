cd ../../
export WANDB_DISABLED=true
deepspeed --include="localhost:1,2,3" --master_port=9902 src/train_bash.py \
    --deepspeed /home/wangzj/temp_data/LLaMA-Factory/llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /home/nfs04/wangzj/models/Qwen1.5-1.8B \
    --adapter_name_or_path /home/nfs02/wangzj/aliyun/1.8b-elhutr8b-4expert-top2-lr5e-5-load_balance_0.01_polar0.01 \
    --train_only_router \
    --ce_loss_coef 0.1 \
    --do_train \
    --dataset slimpajam_1b,tr_1b,el_1b \
    --max_samples 50000 \
    --preprocessing_num_workers 16 \
    --cutoff_len 512 \
    --finetuning_type moe \
    --output_dir /home/nfs04/wangzj/checkpoints/moe/test \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 100 \
    --save_steps 10000 \
    --save_only_model \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16