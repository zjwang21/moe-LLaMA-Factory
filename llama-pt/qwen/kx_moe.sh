#!/bin/bash
set +x
export NCCL_DEBUG=WARN
export NCCL_IB_QPS_PER_CONNECTION=32
export NVTE_APPLY_QK_LAYER_SCALING=0
PYTHON=$(which python)
sed -i "s|barrier_timeout: float = 300|barrier_timeout: float = 1800|g" $PYTHON/site-packages/torch/distributed/elastic/utils/store.py
ENV=dlc
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0,1,2,3 #kexin changed
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4 #kexin changed
elif [ $ENV = dlc ]; then
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
## pip install transformers==4.37.0 -i https://pypi.doubanio.com/simple/
## cd /mnt/cpfs/zhouh/new_code/transformers
## pip install -e ./
cd /mnt/cpfs/zhouh/new_code/LLaMA-Factory/
export WANDB_DISABLED=true
deepspeed \
    --include="$MASTER_ADDR:0,1,2,3,4,5,6,7" \
    --master_port=$MASTER_PORT src/train_bash.py \
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
    --moe_num_experts 3 \
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

# cd ../../
# export WANDB_DISABLED=true
# <<<<<<< HEAD
# deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
#     --deepspeed ./llama-pt/config/ds_config.json \
#     --stage pt \
#     --model_name_or_path /home/nfs04/wangzj/models/Qwen1.5-1.8B \
#     --do_train \
#     --flash_attn \
#     --dataset slimpajam_1b \
# =======
# deepspeed --include="localhost:0,1,2,3" --master_port=9902 src/train_bash.py \
#     --deepspeed /home/yangdezhao/ydz_temp/LLaMA-Factory/llama-pt/config/ds_config.json \
#     --stage pt \
#     --model_name_or_path /home/yangdezhao/ydz_temp/models/Qwen1.5-1.8B \
#     --do_train \
#     --flash_attn \
#     --dataset slimpajam_1b \
#     --max_samples 500000 \
# >>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
#     --preprocessing_num_workers 16 \
#     --cutoff_len 1024 \
#     --finetuning_type moe \
#     --moe_every_k_layers 1 \
# <<<<<<< HEAD
# 	--moe_router_type top1 \
# 	--moe_num_experts 2 \
#     --save_all_params \
# 	--output_dir /home/nfs04/wangzj/checkpoints/kc/qwen4b-moe \
# =======
#     --moe_router_type top1 \
#     --moe_num_experts 2 \
#     --save_all_params \
#     --output_dir /home/yangdezhao/ydz_temp/checkpoints/moe/qwen/qwen-moe \
# >>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
#     --overwrite_output_dir \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_total_limit 1 \
#     --save_only_model \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1.0 \
#     --plot_loss \
# <<<<<<< HEAD
#     --fp16
# =======
#     --bf16
# >>>>>>> 2b1c369609cdc6ca97b0226266289661212e88de
