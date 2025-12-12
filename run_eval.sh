#!/bin/bash

# 任务名称和输出路径
JOB_NAME='eval_latest'
OUTPUT_DIR="eval_results/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"

# 数据集路径配置 (请根据实际情况修改)
PREFIX='data/eval_data/CogVideoX-5B_human_dim'
LABEL_PATH='data/eval_data/CogVideoX-5B_human_dim'

# 待评估的模型权重路径 (请修改为你训练好的 checkpoint 路径)
# 例如: OUTPUT_DIR 下的 checkpoint-best.pth
CHECKPOINT_PATH='checkpoints/humansam/cls_4/cls_4_checkpoint-latest.pth'

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 启动命令
# --num_processes 2 表示使用 2 张卡
# --finetune 指定要加载的权重
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
