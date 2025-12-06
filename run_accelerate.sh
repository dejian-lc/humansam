JOB_NAME='test_decord_accelerate_single'
OUTPUT_DIR="./results/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='data/train_data/cls_test_82_win_video'
LABEL_PATH='data/train_data/cls_test_82_win_video'
MODEL_PATH='checkpoints/internvideo2_model/internvideo2-L14-k400.bin'
DEPTH_MODEL_PATH='./checkpoints/depth_model/depth_pro.pt'

# 使用 accelerate launch 启动
# 如果是单机多卡，accelerate 会自动检测
# 如果需要指定 GPU 数量，可以使用 --num_processes
# 例如: accelerate launch --num_processes 2 run_finetuning_combine.py ...

accelerate launch --num_processes 2 --mixed_precision fp16 run_finetuning_combine.py \
    --model internvideo2_cat_large_patch14_224 \
    --data_path ${LABEL_PATH} \
    --prefix ${PREFIX} \
    --data_set 'SSV2' \
    --split ',' \
    --nb_classes 2 \
    --finetune ${MODEL_PATH} \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 2 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --num_workers 8 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 100 \
    --lr 2e-5 \
    --drop_path 0.1 \
    --head_drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.75 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --use_decord \
    --depth_checkpoint_path ${DEPTH_MODEL_PATH} \
    --no_return_depth
    # --use_conf
