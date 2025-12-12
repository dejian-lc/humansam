#!/bin/bash

# Basic configuration
# Parent directory to be tested
PARENT_DIR='data/eval_data'
# Model checkpoint path (consistent with run_eval.sh, modify if needed)
CHECKPOINT_PATH='checkpoints/humansam/cls_4/cls_4_checkpoint-latest.pth'
# Total output directory for results
BASE_OUTPUT_DIR="eval_results/cls_4_batch_eval_latest"

# Create summary file
mkdir -p "$BASE_OUTPUT_DIR"
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary.csv"
# Write CSV header
echo "Dataset,Accuracy,AUC" > "$SUMMARY_FILE"

# Initialize variables for calculating mean
total_acc=0
total_auc=0
count=0

echo "Starting batch evaluation..."
echo "Parent Directory: $PARENT_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output Directory: $BASE_OUTPUT_DIR"
echo "------------------------------------------------"

# Iterate through all subfolders in the parent directory
for folder_path in "$PARENT_DIR"/*; do
    if [ -d "$folder_path" ]; then
        folder_name=$(basename "$folder_path")
        echo "Processing dataset: $folder_name"
        
        # Set independent output directory and log directory for each dataset
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${folder_name}"
        LOG_DIR="./logs/batch_eval/${folder_name}"
        
        mkdir -p "$OUTPUT_DIR"
        mkdir -p "$LOG_DIR"
        
        # Run evaluation command
        # Use the same parameters as run_eval.sh, dynamically replace data_path and prefix
        accelerate launch --num_processes 2 --mixed_precision fp16 eval.py \
            --model internvideo2_cat_large_patch14_224 \
            --data_path "$folder_path" \
            --prefix "$folder_path" \
            --data_set 'SSV2' \
            --split ',' \
            --nb_classes 4 \
            --finetune "$CHECKPOINT_PATH" \
            --log_dir "$LOG_DIR" \
            --output_dir "$OUTPUT_DIR" \
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
        
        # Extract results
        METRIC_FILE="${OUTPUT_DIR}/metrics.txt"
        if [ -f "$METRIC_FILE" ]; then
            # Parse Accuracy and AUC
            # Assume metrics.txt format: Metric Value (Tab or space separated)
            acc=$(grep "Accuracy" "$METRIC_FILE" | awk '{print $2}')
            auc=$(grep "AUC" "$METRIC_FILE" | awk '{print $2}')
            
            echo "  -> Accuracy: $acc"
            echo "  -> AUC: $auc"
            
            # Write to summary file
            echo "$folder_name,$acc,$auc" >> "$SUMMARY_FILE"
            
            # Accumulate (use python to handle floating point numbers)
            total_acc=$(python3 -c "print($total_acc + $acc)")
            total_auc=$(python3 -c "print($total_auc + $auc)")
            count=$((count + 1))
        else
            echo "  -> Warning: metrics.txt not found in $OUTPUT_DIR"
            echo "$folder_name,N/A,N/A" >> "$SUMMARY_FILE"
        fi
        
        echo "------------------------------------------------"
    fi
done

# Calculate and output mean
if [ $count -gt 0 ]; then
    avg_acc=$(python3 -c "print(f'{$total_acc / $count:.4f}')")
    avg_auc=$(python3 -c "print(f'{$total_auc / $count:.4f}')")
    
    echo "Batch Evaluation Completed."
    echo "Processed $count datasets."
    echo "Average Accuracy: $avg_acc"
    echo "Average AUC: $avg_auc"
    
    # Write mean to summary file
    echo "AVERAGE,$avg_acc,$avg_auc" >> "$SUMMARY_FILE"
    echo "Summary saved to $SUMMARY_FILE"
else
    echo "No datasets were processed. Please check the PARENT_DIR path."
fi
