
# # DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 /SMART_mllab/train.py \
    --output_dir /data/ckpt/ \
    --prediction_type answerkey \
    --SAM_token_mask true \
    --token_mask_path /data/SAM_features/decoder_features/token_mask_features \
    --model_type instructblip_flant5 \
    --pretrained_model_path Salesforce/instructblip-flan-t5-xl \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 25 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 20 \
    --pretrained_module_lr 1e-6 \
    --scratch_module_lr 1e-4 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 8 \
    --project_name SMART_challenge \
    --run_name instructblip_flant5_xl_tokenmask_answerkey \
    --report_to wandb




# # # single gpu run script
# export CUDA_VISIBLE_DEVICES=3
# python /SMART_mllab/train.py \
#     --output_dir /data/ckpt/ \
#     --prediction_type answerkey \
#     --model_type instructblip_flant5 \
#     --SAM_token_mask true \
#     --token_mask_path /data/SAM_features/decoder_features/token_mask_features \
#     --pretrained_model_path Salesforce/instructblip-flan-t5-xxl \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 5 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy steps \
#     --eval_steps 2 \
#     --save_strategy no \
#     --save_steps 1 \
#     --save_total_limit 20 \
#     --pretrained_module_lr 1e-6 \
#     --scratch_module_lr 1e-4 \
#     --warmup_ratio 0.1 \
#     --logging_steps 1 \
#     --lr_scheduler_type cosine \
#     --dataloader_num_workers 0 \
#     --project_name SMART_challenge \
#     --run_name instructblip_baseline \
#     --report_to none



#report_to : ["none", "wandb", ..]
#save_strategy : ["steps", "epochs", "no"]
#model_type : ["instructblip_vicuna", "instructblip_flant5"]
#pretrained_model_path : ["Salesforce/instructblip-vicuna-7b","Salesforce/instructblip-flan-t5-xxl"]
#prediction_type : ["onlyanswer", "answervalue"]

# pretrained_module_lr -> learning rate for pretrained modules (qformer, projection layer)
# scratch_module_lr => learning rate for training from scratch (llm's lora)