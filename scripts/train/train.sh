
# # DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 /SMART_mllab/train.py \
    --output_dir /data/ckpt \
    --num_train_epochs 10 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 20 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 8 \
    --project_name SMART_challenge \
    --run_name instructblip_baseline \
    --report_to wandb


# # single gpu run script
# export CUDA_VISIBLE_DEVICES=0
# python /SMART_mllab/train.py \
#     --output_dir /data/ckpt \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 10 \
#     --per_device_eval_batch_size 5 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy steps \
#     --eval_steps 1 \
#     --save_strategy no \
#     --save_steps 1500 \
#     --save_total_limit 20 \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.1 \
#     --logging_steps 1 \
#     --lr_scheduler_type cosine \
#     --dataloader_num_workers 0 \
#     --project_name SMART_challenge \
#     --run_name instructblip_baseline \
#     --report_to none

