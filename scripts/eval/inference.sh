# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0


python /SeqMMLearning/inference.py \
    --output_dir /data/MOMA/test_ckpt \
    --load_ckpt_path /data/MOMA/test_ckpt \
    --per_device_eval_batch_size 8 \
    --do_predict True \
    --dataloader_num_workers 4 \
    --report_to none
    