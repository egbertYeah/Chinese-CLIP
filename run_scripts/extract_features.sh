#!/bin/bash

# Usage: extract image and text features
# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

SPLIT=${2}
DATAPATH=${3}
RESUME=${DATAPATH}/pretrained_weights/clip_cn_rn50.pt
DATASET_NAME=Flickr30k-CN

python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${DATASET_NAME}/lmdb/${SPLIT}/imgs" \
    --text-data="${DATAPATH}/datasets/${DATASET_NAME}/${SPLIT}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${RESUME} \
    --vision-model=RN50 \
    --text-model=RBT3-chinese
