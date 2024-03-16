# SeqMMLearning
Base code for 2024 NIPS sequential multimodal learning official implementation (or SMART 101 challenge)

+ In order to quickly adapt to the code, this version is for video classification task on MOMA dataset.
+ Thus below guidelines are not for SMART 101 dataset nor IKEA assembly dataset.
+ If you have any problems, feel free to ask Jinwoo.

  


## Setting environment
1. Create docker container at 72 server with CUDA version >= 11.6.
```bash
docker run -it --gpus '"device=0,1,2,3"' --ipc=host --name {your_container_name} -v /data:/media/data2/CATER/
```


2. Clone this repository and create conda environment
```bash
git clone --branch {branch_name} git@github.com:wasabipretzel/SeqMMLearning.git
cd SeqMMLearning
conda env create -f seqmm.yaml conda activate seqmm
```

## Quick start

### Training
+ Below code includes training and evaluation
```bash
./script/train/train.sh
```

+ To run with single gpu, change code in train.sh using comment
```bash 
#train.sh
export CUDA_VISIBLE_DEVICES=0
python /SeqMMLearning/train.py \
...
```


### Inference
```bash
./script/eval/inference.sh
```