#!/bin/bash

# Usage: eval Recall
# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

SPLIT=${2}
DATAPATH=${3}
RESUME=${DATAPATH}/pretrained_weights/clip_cn_rn50.pt
DATASET_NAME=Flickr30k-CN

python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${DATASET_NAME}/${SPLIT}_texts.jsonl \
    ${DATAPATH}/datasets/${DATASET_NAME}/${SPLIT}_predictions.jsonl \
    output.json