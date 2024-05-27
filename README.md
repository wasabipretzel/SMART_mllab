# SMART CHALLENGE 2024 CVPR MAR WORKSHOP
Baseline code for 2024 CVPR SMART 101 challenge

+ Followings are guideline for basic environment setup, training and inference. 
+ Baseline model is Instructblip Vicuna-7b with Lora adapter.
+ If you have any problems, feel free to ask Jinwoo Ahn(jinwooahn@hanyang.ac.kr) or raise PR.


## 🗓️ Schedule
- [X] Modify Vicuna input code considering left padding
- [X] Add multiple learning rate optimizer
- [X] Add pipeline for submission format
- [X] Add function which create save folder when training
- [X] Save generation config
- [X] Add starter code inside this repo and make compatible with starter dataset/models
- [X] Check inference time (make under 5 minutes with 100 qa pairs)
- [X] Modify README for KT corp

## Setting environment (For KT corp)
1. Download dataset and split from below link
``` bash
set +H
wget "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdW9sQ0tyRVZHOWp2Mzd6eDNoZnFXNHFKTFZwP2U9Y2VjNDFO/root/content" -O data.tar
tar -xvf data.tar
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


## Setting environment (For mllab students)
1. Create docker container at 230 server or 72 server with CUDA version >= 11.6.
```bash
docker run -it --gpus '"device=0,1,2,3"' --ipc=host --name {your_container_name} -v /data:/media/data2/SMART101/ {image_id}
```

2. Install conda environment
```bash
apt-get update
apt-get install wget, git, vim
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
vim ~/.bashrc
export PATH="/root/anaconda3/bin:$PATH"
source ~/.bashrc
conda init
```

3. Clone this repository and create conda environment
```bash
git clone git@github.com:wasabipretzel/SMART_mllab.git
cd SMART_mllab
conda env create -f smart_mllab.yaml 
conda activate smart_mllab 
```

4. If conda yaml not working well, start with requirements.txt
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


## Challenge Submission!!

### Step by step
1. docker hub (jinwooahn/hyu_mllab_kt:1.0) 에서 baseline이 되는 image을 받아온다
    - 이 image을 우선 container로 build한 후, SMART_mllab/ 코드 폴더를 최신으로 바꿔준다.
    - /submission_ckpt_dir 아래에 inference에 사용할 ckpt 폴더를 넣어준다

2. Dockerfile을 사용해 image로 build해준다
    - docker file 내용은 아래 참조

3. docker run 을 사용하여 submission.json이 실제로 생성되는지 확인한 후, 각자 docker hub에 올려준다.

4. 제출 Google form에 run 하는 명령어와 함께 제출한다. 


### Step by step (code)

```bash
docker pull jinwooahn/hyu_mllab_kt:1.0

docker build . --file {Dockerfile name} -t jinwooahn/hyu_mllab_kt:1.5

docker run --network none -it --gpus '"device=0"' --ipc=host -v {local test dataset folder}:/SMART_mllab/datasets jinwooahn/hyu_mllab_kt:1.5
```

### Dockerfile
```bash
FROM jinwooahn/hyu_mllab_kt:1.0

RUN pip install -r /SMART_mllab/requirements.txt

RUN echo "pip installed finished"

CMD ["bash", "-c", "cd /SMART_mllab && ./scripts/submission/create_submission.sh"]
```


e.g)
```bash
docker pull jinwooahn/hyu_mllab_kt:1.0


docker build . --file smart_101_submission -t jinwooahn/hyu_mllab_kt:1.5

docker run --network none -it --gpus '"device=2"' --ipc=host -v /media/data2/SMART101/datasets/:/SMART_mllab/datasets jinwooahn/hyu_mllab_kt:1.5
```
