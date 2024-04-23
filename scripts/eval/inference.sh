
# export CUDA_VISIBLE_DEVICES=1
# python /SMART_mllab/inference.py \
#     --output_dir /data/ckpt \
#     --model_type instructblip_flant5 \
#     --prediction_type answerkey \
#     --load_ckpt_path /data/ckpt/instructblip_flant5/onlyanswer/instructblip_baseline_flant5/checkpoint-4000 \
#     --per_device_eval_batch_size 5 \
#     --do_predict True \
#     --dataloader_num_workers 4 \
#     --report_to none


export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 /SMART_mllab/inference.py \
    --output_dir /data/ckpt \
    --model_type instructblip_flant5 \
    --prediction_type answerkey \
    --load_ckpt_path /data/ckpt/instructblip_flant5/onlyanswer/instructblip_baseline_flant5/checkpoint-2000 \
    --per_device_eval_batch_size 5 \
    --do_predict True \
    --dataloader_num_workers 8 \
    --report_to none