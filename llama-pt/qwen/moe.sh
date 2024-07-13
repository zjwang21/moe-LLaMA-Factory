cd ../../
export WANDB_DISABLED=true
<<<<<<< HEAD
deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
    --deepspeed ./llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /home/nfs04/wangzj/models/Qwen1.5-1.8B \
    --do_train \
    --flash_attn \
    --dataset slimpajam_1b \
=======
deepspeed --include="localhost:0,1,2,3" --master_port=9902 src/train_bash.py \
    --deepspeed /home/yangdezhao/ydz_temp/LLaMA-Factory/llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /home/yangdezhao/ydz_temp/models/Qwen1.5-1.8B \
    --do_train \
    --flash_attn \
    --dataset slimpajam_1b \
    --max_samples 500000 \
>>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
    --preprocessing_num_workers 16 \
    --cutoff_len 1024 \
    --finetuning_type moe \
    --moe_every_k_layers 1 \
<<<<<<< HEAD
	--moe_router_type top1 \
	--moe_num_experts 2 \
    --save_all_params \
	--output_dir /home/nfs04/wangzj/checkpoints/kc/qwen4b-moe \
=======
    --moe_router_type top1 \
    --moe_num_experts 2 \
    --save_all_params \
    --output_dir /home/yangdezhao/ydz_temp/checkpoints/moe/qwen/qwen-moe \
>>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
<<<<<<< HEAD
    --fp16
=======
    --bf16
>>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
