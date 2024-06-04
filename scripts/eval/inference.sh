
# export CUDA_VISIBLE_DEVICES=1
# python /SMART_mllab/inference.py \
#     --output_dir /data/ckpt \
#     --model_type instructblip_flant5 \
#     --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
#     --prediction_type answerkey \
#     --load_ckpt_path /data/kt_ckpt/checkpoint-14000 \
#     --per_device_eval_batch_size 10 \
#     --do_predict True \
#     --dataloader_num_workers 0 \
#     --report_to none


export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 /SMART_mllab/inference.py \
    --output_dir /data/ckpt \
    --model_type instructblip_flant5 \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xl \
    --prediction_type answerkey \
    --load_ckpt_path /data/kt_ckpt/checkpoint-14000 \
    --per_device_eval_batch_size 25 \
    --do_predict True \
    --dataloader_num_workers 8 \
    --report_to none