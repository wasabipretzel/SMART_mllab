export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

/opt/conda/envs/videollava/bin/deepspeed /SeqMMLearning/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 8 \
    --mm_projector_lr 1e-6 \
    --qformer_lr 1e-6 \
    --deepspeed /SeqMMLearning/llava/zero3.json \
    --model_name_or_path /SeqMMLearning/checkpoints/llava-v1.5-7b \
    --mm_projector_model_path /SeqMMLearning/checkpoints/llava-v1.5-7b/mm_projector.bin \
    --pretrained_qformer_path /data/pretrained_models/qformer_pretrained \
    --pretrained_qformer_tokenizer_path /data/pretrained_models/qformer_pretrained/qformer_tokenizer \
    --use_pretrained_qformer True \
    --freeze_pretrained True \
    --train_txt_path /data/dataset/split/train.txt \
    --val_txt_path /data/dataset/split/val.txt \
    --feature_path /data/dataset/features \
    --use_qformer True \
    --query_num 32 \
    --version sequential_reasoning \
    --data_path /data/dataset/split/data.json \
    --image_folder /data/data2/khahn/LLaVA/playground/data \
    --vision_tower None \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /data/cache_ckpt \
    --num_train_epochs 100 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_strategy no \
    --save_steps 1500 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb


#qformer_lr은 use_pretrained_qformer을 쓰는 경우에만 넣기 None이 아니면 lr다르게 들어가게 해놨음.

#pretrained qformer쓰려면 query 개수 32개여야함