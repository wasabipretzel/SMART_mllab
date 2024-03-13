# export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node 1 
    # --deepspeed /SeqMMLearning/llava/zero3.json \
/opt/conda/envs/llava/bin/deepspeed --include localhost:2 /SeqMMLearning/llava/train/train.py \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --mm_projector_lr 1e-6 \
    --freeze_pretrained True \
    --bf16 True \
    --output_dir /data/dataset/cache_ckpt \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 70 \
    --save_strategy no \
    --save_steps 1500 \
    --save_total_limit 20 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none


#qformer_lr은 use_pretrained_qformer을 쓰는 경우에만 넣기 None이 아니면 lr다르게 들어가게 해놨음.

#pretrained qformer쓰려면 query 개수 32개여야함