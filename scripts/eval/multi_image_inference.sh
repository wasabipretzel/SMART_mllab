export CUDA_VISIBLE_DEVICES=2

# --image-file /data/IKEA/dataset/Furniture/Beds/90371954/step \
# --image-file /data/IKEA/dataset/Furniture/Beds/90371954/manual/1 \
python -m llava.eval.run_llava \
    --model-path /LLaVA/checkpoints/llava-v1.5-7b \
    --image-file "/data/IKEA/dataset/Furniture/Beds/90371954/manual/1/page-1.png,/data/IKEA/dataset/Furniture/Beds/90371954/manual/1/page-4.png" \
    --query "<image-placeholder> <image-placeholder> The following images are instructional images to assist with assembling furniture. Page numbers are in the bottom left or bottom right. The bold numbers displayed in the upper left corner are the assembly order. Question : Explain detail description of step 1 (page 4)" \

