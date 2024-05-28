
export CUDA_VISIBLE_DEVICES=0
# to handle error when using in KT server(just in case)
export PATH=$PATH:path/to/HIP/bin

python /home/work/instructblip/SMART_mllab/inference.py \
    --output_dir /home/work/instructblip/data/ckpt \
    --model_type instructblip_flant5 \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xl \
    --prediction_type answerkey \
    --load_ckpt_path /home/work/instructblip/data/ckpt/instructblip_flant5/answerkey/instructblip_baseline_flant5_answerkey/checkpoint-6000 \
    --per_device_eval_batch_size 20 \
    --do_predict True \
    --dataloader_num_workers 4 \
    --report_to wandb


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