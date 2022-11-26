#!/bin/bash

# Usage: topK search
# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

SPLIT=${2}
DATAPATH=${3}
RESUME=${DATAPATH}/pretrained_weights/clip_cn_rn50.pt
DATASET_NAME=Flickr30k-CN

python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${DATASET_NAME}/${SPLIT}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${DATASET_NAME}/${SPLIT}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${DATASET_NAME}/${SPLIT}_predictions.jsonl"
