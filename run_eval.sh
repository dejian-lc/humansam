#!/bin/bash

# Task name and output path
JOB_NAME='eval_latest'
OUTPUT_DIR="eval_results/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"

# Dataset path configuration (please modify according to actual situation)
PREFIX='data/eval_data/CogVideoX-5B_human_dim'
LABEL_PATH='data/eval_data/CogVideoX-5B_human_dim'

# Model checkpoint path for evaluation (please modify to your trained checkpoint path)
# Example: checkpoint-best.pth under OUTPUT_DIR
CHECKPOINT_PATH='checkpoints/humansam/cls_4/cls_4_checkpoint-latest.pth'

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Launch command
# --num_processes 2 means using 2 GPUs
# --finetune specifies the weight to load
accelerate launch --num_processes 2 --mixed_precision fp16 eval.py \
    --model internvideo2_cat_large_patch14_224 \
    --data_path ${LABEL_PATH} \
    --prefix ${PREFIX} \
    --data_set 'SSV2' \
    --split ',' \
    --nb_classes 2 \
    --finetune ${CHECKPOINT_PATH} \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_frames 8 \
    --sampling_rate 4 \
    --num_sample 1 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_workers 8 \
    --warmup_epochs 0 \
    --tubelet_size 1 \
    --opt adamw \
    --lr 1e-3 \
    --drop_path 0.1 \
    --head_drop_path 0.0 \
    --layer_decay 0.9 \
    --opt_betas 0.9 0.999 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --use_decord \
    # --no_return_depth
