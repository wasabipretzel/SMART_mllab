
export CUDA_VISIBLE_DEVICES=0
python /SMART_mllab/submission.py \
    --output_dir /data/ckpt \
    --model_type instructblip_flant5 \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xl \
    --challenge_phase test \
    --per_device_eval_batch_size 1 \
    --do_predict True \
    --class_ensemble true \
    --load_key_ckpt_path /submission_ckpt_dir/keymodel_ckpt/checkpoint-4000 \
    --load_value_ckpt_path /submission_ckpt_dir/valuemodel_ckpt/checkpoint-4000 \
    --prediction_type ensemble_classify_category \
    --use_dynamic_sam_decoder true \
    --sam_pretrained_model_path /submission_ckpt_dir/sam-vit-huge \
    --qwen_pretrained_model_path /submission_ckpt_dir/Qwen-VL-Chat \
    --dataloader_num_workers 0 \
    --report_to none

# --use_dynamic_caption true \


# export CUDA_VISIBLE_DEVICES=1,2,3
# torchrun --nproc_per_node 3 /SMART_mllab/inference.py \
#     --output_dir /data/ckpt \
#     --model_type instructblip_flant5 \
#     --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
#     --prediction_type answerkey \
#     --challenge_phase val \
#     --load_ckpt_path /data/ckpt/instructblip_flant5/answerkey/instructblip_baseline_flant5_answerkey/checkpoint-6000 \
#     --per_device_eval_batch_size 5 \
#     --do_predict True \
#     --dataloader_num_workers 8 \
#     --report_to none
