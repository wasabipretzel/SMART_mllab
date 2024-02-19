"""
    utility functions for parsing additional context from ASM / IAW dataset annotation
"""
import os
import json 



def ASM_action_annotation(furniture):
    with open('/SeqMMLearning/context_materials/ASM/action_annotations/gt_segments.json', 'r') as f:
        annot = json.load(f)["database"]

    furnitures = []
    for k in annot.keys():
        furnitures.append(k.split('/')[0])
    furnitures = list(set(furnitures))
    # ['Lack_TV_Bench', 'Lack_Side_Table', 'Lack_Coffee_Table', 'Kallax_Shelf_Drawer']

    for k, v in annot.items():
        if furniture in k:
            actions = []
            for action_dummy in v["annotation"]:
                if action_dummy['label'] != 'NA' and action_dummy['label'] != 'other':
                    actions.append(action_dummy['label'])
            action_context = ""
            for idx, act in enumerate(actions):
                action_context += f"{idx+1}. {act} \n "
            breakpoint()
    return 







if __name__ == "__main__":
    # ['Lack_TV_Bench', 'Lack_Side_Table', 'Lack_Coffee_Table', 'Kallax_Shelf_Drawer']
    furniture = "Lack_TV_Bench"
    ASM_action_annotation(furniture)