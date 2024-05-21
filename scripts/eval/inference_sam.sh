
# export CUDA_VISIBLE_DEVICES=1
# python /SMART_mllab/inference.py \
#     --output_dir /data/ckpt \
#     --model_type instructblip_flant5 \
#     --use_SAM true \
#     --sam_feature_path /data/SAM_features/encoder_features/avg_pooled_features \
#     --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
#     --prediction_type answervalue \
#     --load_ckpt_path /data/ckpt/instructblip_flant5/answervalue/instructblip_flant5_SAMCONCAT_answervalue/checkpoint-6000 \
#     --per_device_eval_batch_size 10 \
#     --do_predict True \
#     --dataloader_num_workers 4 \
#     --report_to none


export CUDA_VISIBLE_DEVICES=0,1,3
torchrun --nproc_per_node 3 /SMART_mllab/inference.py \
    --output_dir /data/ckpt \
    --model_type instructblip_flant5 \
    --sam_feature_path /data/SAM_features/encoder_features/avg_pooled_features \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
    --prediction_type answervalue \
    --load_ckpt_path /data/ckpt/instructblip_flant5/answervalue/instructblip_flant5_SAMCONCAT_answervalue/checkpoint-10000 \
    --per_device_eval_batch_size 10 \
    --do_predict True \
    --dataloader_num_workers 8 \
    --report_to none