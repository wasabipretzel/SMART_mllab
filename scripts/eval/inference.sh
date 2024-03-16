# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node 1 
    # --deepspeed /SeqMMLearning/llava/zero3.json \

# torchrun --nproc_per_node 4 /SeqMMLearning/train.py \
python /SeqMMLearning/inference.py \
    --output_dir /data/MOMA/vivit_ckpt \
    --load_ckpt_path /data/MOMA/vivit_ckpt \
    --per_device_eval_batch_size 1 \
    --do_predict True \
    --dataloader_num_workers 4 \
    --project_name moma_vivit \
    --run_name trial1 \
    --report_to none
    