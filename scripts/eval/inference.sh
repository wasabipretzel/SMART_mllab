
export CUDA_VISIBLE_DEVICES=0
python /SMART_mllab/inference.py \
    --output_dir /data/ckpt \
    --load_ckpt_path /data/ckpt/checkpoint-10500 \
    --per_device_eval_batch_size 3 \
    --do_predict True \
    --dataloader_num_workers 8 \
    --report_to none


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node 4 /SMART_mllab/inference.py \
#     --output_dir /data/ckpt \
#     --load_ckpt_path /data/ckpt/checkpoint-10500 \
#     --per_device_eval_batch_size 5 \
#     --do_predict True \
#     --dataloader_num_workers 8 \
#     --report_to none