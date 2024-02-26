python generate_instruct_qa.py \
        --category_type='1' \
        --query_list="/data/dataset/mapping/vid_valid_mapping.json" \
        --fewshot_path='/SeqMMLearning/instruct_generation/fewshot/single_turn' \
        --sys_msg_path="/SeqMMLearning/instruct_generation/system_msg.json" \
        --annot_path="/data/generate/merged_0_280_data.json" \
        --manual_path="/data/dataset/manuals" \
        --save_path="/data/generate/qa_wo_annot/single_turn/" \
        --mode='wo_annot' \
        --api_key="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" \

# python generate_instruct_qa.py \
#         --category_type='2' \
#         --query_list="/data/dataset/mapping/vid_valid_mapping.json" \
#         --fewshot_path='/SeqMMLearning/instruct_generation/fewshot/single_turn' \
#         --sys_msg_path="/SeqMMLearning/instruct_generation/system_msg.json" \
#         --annot_path="/data/generate/merged_0_280_data.json" \
#         --manual_path="/data/dataset/manuals" \
#         --save_path="/data/generate/qa_wo_annot/single_turn/" \
#         --mode='wo_annot' \
#         --api_key="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" \

# python generate_instruct_qa.py \
#         --category_type='3' \
#         --query_list="/data/dataset/mapping/vid_valid_mapping.json" \
#         --fewshot_path='/SeqMMLearning/instruct_generation/fewshot/single_turn' \
#         --sys_msg_path="/SeqMMLearning/instruct_generation/system_msg.json" \
#         --annot_path="/data/generate/merged_0_280_data.json" \
#         --manual_path="/data/dataset/manuals" \
#         --save_path="/data/generate/qa_wo_annot/single_turn/" \
#         --mode='wo_annot' \
#         --api_key="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" \