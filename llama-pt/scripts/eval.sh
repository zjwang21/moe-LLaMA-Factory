export CUDA_VISIBLE_DEVICES=3
python /home/wangzj/LLaMA-Factory/src/evaluate.py \
    --model_name_or_path /home/nfs01/llama/model/llama-hf/llama-7b-hf \
    --checkpoint_dir /home/nfs02/wangzj/checkpoints/llama2-lora-alpaca-zh-en-5e6 \
    --finetuning_type lora \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 2 \
    --task_dir /home/wangzj/LLaMA-Factory/evaluation \