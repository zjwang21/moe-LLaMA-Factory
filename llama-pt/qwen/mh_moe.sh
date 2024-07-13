cd ../../
export WANDB_DISABLED=true
# export WANDB_PROJECT=mh_moe_ardeishi8b
deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
    --deepspeed ./llama-pt/config/ds_config_cpu.json \
    --stage pt \
    --model_name_or_path /home/nfs04/wangzj/models/Qwen1.5-1.8B \
    --do_train \
    --flash_attn \
    --dataset ar_2b,de_2b,is_2b,hi_2b \
    --mix_strategy concat \
    --preprocessing_num_workers 32 \
    --cache_path /home/nfs03/wangzj/dataset/pretrain/arderu6b \
    --cutoff_len 1024 \
    --finetuning_type moe \
    --moe_every_k_layers 1 \
	--moe_router_type mh_moe \
	--moe_num_experts 4 \
    --moe_num_heads 4 \
    --topk 2 \
    --moe_with_aux \
	--output_dir /home/nfs04/wangzj/checkpoints/moe/1.8b-arderu-mhmoe \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 1 \
    --save_only_model \
    --save_steps 10000 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16
