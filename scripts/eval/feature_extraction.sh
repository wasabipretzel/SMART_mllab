export CUDA_VISIBLE_DEVICES=1


# root@766c8cea6e97:/SeqMMLearning# ./scripts/eval/feature_extraction.sh 
python -m llava.eval.get_feature \
    --model-path /SeqMMLearning/checkpoints/llava-v1.5-7b \
    --image_folder "/data/dataset/manuals" \
    --output_folder "/data/dataset/features" 