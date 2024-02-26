import os
import shutil
import json

def generate_mapping():
    """
        downscaled된 비디오가 존재하는 serial번호만 취급
        {
            serial_num : {
                'manual_name'
                'vid_name'
                'vid_duration'
                'vid_fps'
                'vid_annotation'
                'subcategory'
                'type_name'
                'youtube_script'
            }
        }
    """
    with open('/data/dataset/mapping/manual_mapping.json', 'r') as f:
        manual_serial_map = json.load(f)
    with open('/data/dataset/mapping/num2vid_mapping_renewal.json', 'r') as f:
        num2vid_map = json.load(f)
    with open('/data/dataset/mapping/reverse_manual_mapping.json', 'r') as f:
        serial_manual_map = json.load(f)
    with open('/data/dataset/mapping/IKEAAssemblyInTheWildDataset.json', 'r') as f:
        metadata = json.load(f)
    with open('/data/dataset/mapping/prompt_1_2.json', 'r') as f:
        kh_map = json.load(f)
    result_map = {}
    #모든 비디오 list받아오기
    all_vids = os.listdir('/data/dataset/videos')
    #serial 번호 -> 해당하는 video가 downscaled video에 있으면 mapping에 추가
    #serial 번호는 dataset 하위의 모든 번호를 가져온다.
    all_serials = os.listdir('/data/dataset/manuals')
    for each_serial in all_serials:
        #video 존재 여부 판단
        # vid = num2vid_map[each_serial]["min_vid_name"]
        if each_serial in num2vid_map.keys():
            vid = num2vid_map[each_serial]["min_vid_name"]
            manual_name = os.listdir(f'/data/dataset/manuals/{each_serial}/pdf')[0]
            vid_annotation = num2vid_map[each_serial]["min_vid_duration"]
            vid_fps = num2vid_map[each_serial]["min_vid_fps"]

            # vid_duration / name, subCategory, typeName, youtube_script 
            for seg in kh_map:
                if seg["id"] == each_serial:
                    name = seg["name"]
                    subcategory = seg["subCategory"]
                    typename = seg["typeName"]
                    youtube_script = seg["youtube_script"]
            
            for seg in metadata:
                if seg["id"] == each_serial:
                    vid_list = seg["videoList"]
                    for each_vid_candidate in vid_list:
                        can_vid_name = each_vid_candidate["url"].split('=')[-1]
                        if can_vid_name == vid:
                            vid_duration = each_vid_candidate["duration"]
            result_map[each_serial] = {
                'manual_name' : manual_name,
                'vid_name' : vid,
                'vid_duration' : vid_duration,
                'vid_fps' : vid_fps,
                'vid_annotation' : vid_annotation,
                'subCategory': subcategory,
                'name' : name,
                'typeName' : typename,
                'youtube_script' : youtube_script
            }
    
    with open('/data/dataset/mapping/vid_valid_mapping.json', 'w') as f:
        json.dump(result_map,f)

            

def gen_num2vid_map():
    with open('/data/dataset/mapping/IKEAAssemblyInTheWildDataset.json', 'r') as f:
        metadata = json.load(f)
    all_serials = os.listdir('/data/dataset/manuals')
    all_vids = os.listdir('/data/dataset/videos')
    #vid가 있으면서 제일 짧은 애를 pick

    result = {}
    for each_serial in all_serials:
        for seg in metadata:
            if seg["id"] == each_serial:
                #vidlist찾기
                vid_list = seg["videoList"]
                min_dur = 10000000000000000
                min_vid = ""
                min_annot=""
                min_fps=0
                for vid in vid_list:
                    vid_name = vid["url"].split("=")[-1]
                    if vid_name+'.mp4' in all_vids:
                        # breakpoint()
                        if vid["duration"] < min_dur:
                            min_dur = vid["duration"]
                            min_vid = vid_name
                            min_annot = vid["annotation"]
                            min_fps = vid["fps"]
                            min_dur = vid["duration"]
                if min_dur != 10000000000000000: #vid없는경우
                    result[each_serial] = {
                        "min_vid_name" : min_vid,
                        "min_vid_duration" : min_annot,
                        "min_vid_fps" : min_fps,
                        "min_vid_dur" : min_dur
                    }
    with open('/data/dataset/mapping/num2vid_mapping_renewal.json', 'w') as f:
        json.dump(result, f)
    return

def split_list(lst, segment_size):
    return [lst[i:i+segment_size] for i in range(0, len(lst), segment_size)]

def save_list_to_txt(lst, file_path):
    with open(file_path, 'w') as f:
        for item in lst:
            f.write(str(item) + '\n')

def split_manuals_into_api_num():
    # get all manuals
    with open('/data/dataset/mapping/vid_valid_mapping.json', 'r') as f:
        all_manual_data = json.load(f)
    
    all_manuals = list(all_manual_data.keys())

    # breakpoint()
    segment_size = len(all_manuals) // 7
    segments = split_list(all_manuals, segment_size) #[[12323, 121212,..], [121212,31232,..]..]

    # breakpoint()
    with open('/data/dataset/mapping/openai_api_genlist.json', 'r') as f:
        openai_genlist = json.load(f)
    
    for idx, seg in enumerate(segments):
        openai_genlist[f"API_{idx+1}"]["manuals"] = seg
    
    # NOTE 잘못 돌릴 경우를 대비해 주석처리
    # with open('/data/dataset/mapping/openai_api_genlist.json', 'w') as f:
    #     json.dump(openai_genlist, f)



def single_api_num():
    with open('/data/dataset/mapping/vid_valid_mapping.json', 'r') as f:
        all_manual_data = json.load(f)
    
    all_manuals = list(all_manual_data.keys())

    result = {}

    result[f"API_1"] = {
        "api" : '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',
        'manuals' : all_manuals
    }
    
    with open('/data/dataset/mapping/single_api_genlist.json', 'w') as f:
        json.dump(result, f)

    return 


def mult_api_same_all_manuals():
    with open('/data/dataset/mapping/vid_valid_mapping.json', 'r') as f:
        all_manual_data = json.load(f)
    
    #get reference api files
    with open('/data/dataset/mapping/openai_api_key_jw.json', 'r') as f:
        api_keys = json.load(f)
    
    keys = []
    for k, v in api_keys.items():
        keys.append(v["api"])
    
    all_manuals = list(all_manual_data.keys())

    result = {}

    for idx, key in enumerate(keys):
        result[f"API_{idx+1}"] = {
            "api" : key,
            "manuals" : all_manuals
        }
    
    with open('/data/dataset/mapping/multiple_api_all_manual_genlist.json', 'w') as f:
        json.dump(result, f)

    return 



if __name__ == '__main__':
    # generate_mapping()
    # gen_num2vid_map()
    # split_manuals_into_api_num()

    # single_api_num()
    # mult_api_same_all_manuals()