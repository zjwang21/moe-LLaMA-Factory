export WANDB_DISABLED=true
cd ../../
deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
    --deepspeed /mnt/cpfs/zhouh/new_code/LLaMA-Factory/llama-pt/config/ds_config.json \
    --stage pt \
    --model_name_or_path /mnt/cpfs/zhouh/models/Qwen/Qwen1.5-1.8B \
    --do_train \
    --flash_attn \
    --dataset ar_2b,de_2b,ru_2b \
    --preprocessing_num_workers 16 \
    --cutoff_len 1024 \
    --finetuning_type moe \
    --moe_every_k_layers 1 \
    --moe_router_type top1 \
    --moe_num_experts 4 \
    --use_load_balancing_loss \
    --router_aux_loss_coef 0.01 \
    --output_dir /mnt/cpfs/zhouh/checkpoints/stage1/kx_1.8b-arderu6b-3experts-top1-lr2e-5 \
    --overwrite_output_dir \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 3 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --use_polarization_loss  \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16