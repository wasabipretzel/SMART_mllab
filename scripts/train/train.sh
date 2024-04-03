
# DDP run script
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node 4 /SeqMM/train.py \
#     --output_dir /data/MOMA/test_ckpt \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 3 \
#     --per_device_eval_batch_size 3 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy steps \
#     --eval_steps 40 \
#     --save_strategy no \
#     --save_steps 1500 \
#     --save_total_limit 20 \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.1 \
#     --logging_steps 1 \
#     --warmup_ratio 0.1 \
#     --lr_scheduler_type cosine \
#     --dataloader_num_workers 8 \
#     --project_name moma_vivit \
#     --run_name trial2 \
#     --report_to none

# single gpu run script
export CUDA_VISIBLE_DEVICES=0
python /SeqMM/train.py \
    --output_dir /data/test_ckpt \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 40 \
    --save_strategy no \
    --save_steps 1500 \
    --save_total_limit 20 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 0 \
    --project_name moma_vivit \
    --run_name trial2 \
    --report_to none

