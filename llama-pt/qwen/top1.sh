cd ../../
export WANDB_DISABLED=true
deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
    --deepspeed /home/wangzj/LLaMA-Factory-hx/llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /home/nfs04/wangzj/models/Qwen1.5-1.8B \
    --adapter_name_or_path /home/nfs04/wangzj/checkpoints/moe/1.8b-arderu6b-top2router-arderuen5w \
    --train_only_router \
    --moe_router_type top1 \
    --do_train \
    --dataset slimpajam_1b,ar_2b,de_2b,ru_2b \
    --max_samples 50000 \
    --preprocessing_num_workers 16 \
    --cutoff_len 512 \
    --finetuning_type moe \
    --output_dir /home/nfs04/wangzj/checkpoints/moe/1.8b-arderu6b-top2router-arderuen5w-top1 \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 100 \
    --save_steps 10000 \
    --save_only_model \
    --learning_rate 2e-4 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16