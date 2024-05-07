
export CUDA_VISIBLE_DEVICES=4,5
python inference.py \
    --output_dir ./data/ckpt/size448 \
    --model_type instructblip_flant5 \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
    --prediction_type answerkey \
    --load_ckpt_path /home/work/g-earth-22/VLM/users/mjkim/SMART-101/SMART_mllab/data/ckpt/size448/instructblip_flant5/answerkey/instructblip_baseline_flant5_size448/checkpoint-70000 \
    --per_device_eval_batch_size 10 \
    --do_predict True \
    --dataloader_num_workers 4 \
    --report_to none


# export CUDA_VISIBLE_DEVICES=1,2,3
# torchrun --nproc_per_node 3 /SMART_mllab/inference.py \
#     --output_dir /data/ckpt \
#     --model_type instructblip_flant5 \
#     --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
#     --prediction_type answerkey \
#     --load_ckpt_path /data/ckpt/instructblip_flant5/answerkey/instructblip_baseline_flant5_answerkey/checkpoint-6000 \
#     --per_device_eval_batch_size 5 \
#     --do_predict True \
#     --dataloader_num_workers 8 \
#     --report_to none
