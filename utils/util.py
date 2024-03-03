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



def merge_response_into_one(): 
    """
        생성한 1~6 카테고리 결과들을 모아
        dataset/split  아래에 data.json 으로 저장
        이 data.json 의 key값들 중 8:2 비율로 random split하여 train, val txt 생성
    """
    cat_num = [1,2,3,4,5,6]
    input_folder = "/data/generate/qa_annot/single_turn"
    save_folder = "/data/dataset/split"
    candidate = os.listdir(input_folder)

    input_jsons = []
    for each_candid in candidate:
        if 'error' not in each_candid:
            input_jsons.append(each_candid)
    
    all_qas = []
    # {id , content} 을 key로 하는 dict들을 append하는 list을 생성
    for each_json in input_jsons:
        each_path = os.path.join(input_folder, each_json)
        with open(each_path, 'r') as f:
            catqa = json.load(f)
        for each_serial in catqa.keys():
            serial_id = each_serial
            msg = catqa[serial_id]["choices"][0]["message"]["content"]
            all_qas.append(
                {
                    "id" : serial_id,
                    "content" : msg
                }
            )

    #all_qas을 enumerate돌면서 result에 등록
    result = {}
    for idx, qadict in enumerate(all_qas):
        result[idx] = qadict

    with open(os.path.join(save_folder, 'data.json'), 'w') as f:
        json.dump(result, f)
    
    #random sample train and val
    import random

    target_list = list(result.keys())
    tot_len = len(target_list)
    train_len  = int(tot_len*0.8)

    train_list = random.sample(target_list, train_len)

    val_list = [item for item in target_list if item not in train_list]

    # Open the file in write mode
    with open(os.path.join(save_folder, 'train.txt'), 'w') as file:
        # Write each item of the list to the file
        for item in train_list:
            file.write('%s\n' % item)

    with open(os.path.join(save_folder, 'val.txt'), 'w') as file:
        # Write each item of the list to the file
        for item in val_list:
            file.write('%s\n' % item)


    return


def split_train_val():
    """
        train : val = 8 : 2
    """
    import random
    
    with open('/data/dataset/split/data.json', 'r') as f:
        data = json.load(f)

    save_path = "/data/dataset/split"

    candidates = list(data.keys())

    random.shuffle(candidates)

    # 8:2 비율로 분할
    split_index = int(len(candidates) * 0.8)
    train_set = candidates[:split_index]
    val_set = candidates[split_index:]

    print("Train set:", len(train_set))
    print("Test set:", len(val_set))

    save_list_to_txt(train_set, os.path.join(save_path, 'train.txt'))
    save_list_to_txt(val_set, os.path.join(save_path, 'val.txt'))



def merge_error_handler_into_one():
    base_path = "/data/generate/qa_annot/error_handler"
    cat_num = [1,3,6]
    input_folder = "/data/generate/qa_annot/error_handler"
    save_folder = "/data/dataset/split"
    candidate = os.listdir(input_folder)

    input_jsons = []
    for each_candid in candidate:
        if 'error' not in each_candid:
            input_jsons.append(each_candid)
    
    all_qas = []
    # {id , content} 을 key로 하는 dict들을 append하는 list을 생성
    for each_json in input_jsons:
        each_path = os.path.join(input_folder, each_json)
        with open(each_path, 'r') as f:
            catqa = json.load(f)
        for each_serial in catqa.keys():
            serial_id = each_serial
            msg = catqa[serial_id]["choices"][0]["message"]["content"]
            all_qas.append(
                {
                    "id" : serial_id,
                    "content" : msg
                }
            ) 

    result = {}
    for idx, qadict in enumerate(all_qas):
        result[idx] = qadict

    with open(os.path.join(save_folder, 'error_merged.json'), 'w') as f:
        json.dump(result, f)

    return


def merge_into_one_data():
    with open('/data/dataset/split/data.json', 'r') as f:
        legacy = json.load(f)
    
    with open('/data/dataset/split/preprocessed_error_data.json', 'r') as f:
        newone = json.load(f)
    
    # key를 새로 만들어야함
    last_key_legacy = list(legacy.keys())[-1]

    start_key = str(int(last_key_legacy)+1)

    #새로운 value들 담아놓기
    new_val = []
    for k, v in newone.items():
        new_val.append(v)

    for val in new_val:
        legacy[start_key] = val
        start_key = str(int(start_key)+1)

    with open('/data/dataset/split/new_data.json', 'w') as f:
        json.dump(legacy, f)

    return



if __name__ == '__main__':
    # generate_mapping()
    # gen_num2vid_map()
    # split_manuals_into_api_num()

    # single_api_num()
    # mult_api_same_all_manuals()
    # merge_response_into_one()
    split_train_val()

    # merge_error_handler_into_one()

    # merge_into_one_data()
