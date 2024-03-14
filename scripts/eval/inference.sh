# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node 1 
    # --deepspeed /SeqMMLearning/llava/zero3.json \

# torchrun --nproc_per_node 4 /SeqMMLearning/train.py \
python /SeqMMLearning/train.py \
    --output_dir /data/MOMA/vivit_ckpt \
    --num_train_epochs 20 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 1 \
    --save_strategy no \
    --save_steps 1500 \
    --save_total_limit 20 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 0 \
    --project_name moma_vivit \
    --run_name trial1 \
    --report_to none
    