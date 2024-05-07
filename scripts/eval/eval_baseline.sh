
export CUDA_VISIBLE_DEVICES=6,7
python inference.py --master_port 56789 \
    --output_dir ./data/ckpt/baseline \
    --model_type instructblip_flant5 \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
    --prediction_type answerkey \
    --load_ckpt_path /home/work/g-earth-22/VLM/users/mjkim/SMART-101/SMART_mllab/data/ckpt/baseline/instructblip_flant5/answerkey/instructblip_baseline_flant5_baseline/checkpoint-6000 \
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