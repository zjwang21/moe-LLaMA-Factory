cd ../../
export WANDB_DISABLED=true
deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
<<<<<<< HEAD
    --deepspeed /home/wangzj/LLaMA-Factory-hx/llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /home/nfs04/wangzj/models/Qwen1.5-1.8B \
    --adapter_name_or_path /home/nfs02/wangzj/checkpoints/llama-moe/qwen/1.8b-arderu6b-moe \
    --train_only_router \
    --moe_router_type top2 \
    --do_train \
    --dataset slimpajam_1b,ar_2b,de_2b,ru_2b \
    --max_samples 50000 \
    --preprocessing_num_workers 16 \
    --cutoff_len 512 \
    --finetuning_type moe \
    --output_dir /home/nfs04/wangzj/checkpoints/moe/1.8b-arderu6b-top2router-arderuen5w \
=======
    --deepspeed /home/yangdezhao/ydz_temp/LLaMA-Factory/llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /home/yangdezhao/ydz_temp/models/Qwen1.5-1.8B \
    --adapter_name_or_path /home/yangdezhao/ydz_temp/checkpoints/moe/qwen/1.8b-arderu6b-moe \
    --init_moe_weights \
    --train_only_router \
    --do_train \
    --flash_attn \
    --dataset slimpajam_1b \
    --max_samples 500000 \
    --preprocessing_num_workers 16 \
    --cutoff_len 1024 \
    --finetuning_type moe \
    --output_dir /home/yangdezhao/ydz_temp/checkpoints/moe/qwen/1.8b-arderu6b-routeren50w-saved10w \
>>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 100 \
<<<<<<< HEAD
    --save_steps 10000 \
    --save_only_model \
    --learning_rate 2e-4 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
=======
    --save_steps 195 \
    --save_only_model \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16
>>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
