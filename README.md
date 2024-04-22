# SMART CHALLENGE 2024 CVPR MAR WORKSHOP
Baseline code for 2024 CVPR SMART 101 challenge

+ Followings are guideline for basic environment setup, training and inference. 
+ Baseline model is Instructblip Vicuna-7b with Lora adapter.
+ If you have any problems, feel free to ask Jinwoo Ahn(jinwooahn@hanyang.ac.kr) or raise PR.


## 🗓️ Schedule
- [X] Modify Vicuna input code considering left padding
- [X] Add multiple learning rate optimizer
- [ ] Add pipeline for submission format
- [X] Add function which create save folder when training
- [X] Save generation config
- [X] Add starter code inside this repo and make compatible with starter dataset/models
- [X] Check inference time (make under 5 minutes with 100 qa pairs)
- [ ] Modify README for KT corp



## Setting environment
1. Create docker container at 230 server with CUDA version >= 11.6. (72 server will be also setup soon..)
```bash
docker run -it --gpus '"device=0,1,2,3"' --ipc=host --name {your_container_name} -v /data:/media/data2/SMART/ 5e7815e32cbc
```


2. Clone this repository and create conda environment
```bash
git clone git@github.com:wasabipretzel/SMART_mllab.git
cd SMART_mllab
conda env create -f smart_mllab.yaml 
conda activate smart_mllab 
```

3. If conda yaml not working well, start with requirements.txt
```bash
git clone git@github.com:wasabipretzel/SMART_mllab.git
cd SMART_mllab
conda create -n smart_mllab python=3.9
conda activate smart_mllab
pip install -r requirements.txt
```

## Quick start

### Training
+ Below shell script support single/multiple gpu training.
```bash
./script/train/train.sh
```

### Inference
```bash
./script/eval/inference.sh
```
