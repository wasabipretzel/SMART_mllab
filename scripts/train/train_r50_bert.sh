# for smart starter model R50+BERT


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node 4 /SMART_mllab/train.py \
#     --model_type R50_BERT \
#     --seed 9512 \
#     --output_dir /data/ckpt/starter_model \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 25 \
#     --per_device_eval_batch_size 25 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy steps \
#     --eval_steps 20 \
#     --save_strategy no \
#     --save_steps 1500 \
#     --save_total_limit 20 \
#     --learning_rate 1e-3 \
#     --warmup_ratio 0.01 \
#     --logging_steps 1 \
#     --lr_scheduler_type cosine \
#     --dataloader_num_workers 8 \
#     --project_name SMART_challenge \
#     --run_name R50_BERT \
#     --predict_with_generate False \
#     --include_inputs_for_metrics True \
#     --report_to none


export CUDA_VISIBLE_DEVICES=3
python /SMART_mllab/train.py \
    --model_type R50_BERT \
    --seed 9512 \
    --output_dir /data/ckpt/starter_model \
    --num_train_epochs 10 \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 100 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy no \
    --save_steps 1500 \
    --save_total_limit 20 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 8 \
    --project_name SMART_challenge \
    --run_name R50_BERT \
    --predict_with_generate False \
    --include_inputs_for_metrics True \
    --report_to wandb
