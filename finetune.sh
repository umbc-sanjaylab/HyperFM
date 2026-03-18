#!/bin/bash
LOGFILE="finetune_dir/$(date +'%Y%m%d_%H%M%S').log"

# Save both stdout and stderr to the log file, while still printing to console
exec > >(tee -a "$LOGFILE") 2>&1
echo "--- Checking GPUs with nvidia-smi ---"
nvidia-smi

echo "--- Checking GPU memory explicitly ---"
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader

# Explicitly set visible GPUs to GPU 0 and GPU 1
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc-per-node=4  --rdzv_backend=c10d \
--rdzv_endpoint=localhost:$(shuf -i 20000-30000 -n 1) \
main_finetune.py \
--batch_size 16 --accum_iter 16 \
--epochs 250 --warmup_epochs 20 --patience 30 \
--input_size 96 --patch_size 8 \
--model_type hyperFM_conv \
--model hyperFM_enc4 --nb_classes 4 \
--dataset_type pace \
--data_mode all \
--blr 0.001 --num_workers 6 \
--train_path preprocessed_data/preprocessed_data_list.csv \
--val_path preprocessed_data/preprocessed_data_list.csv \
--test_path preprocessed_data/preprocessed_data_list.csv \
--output_dir ./finetune_dir \
--log_dir ./finetune_dir \
--finetune_dec_only \
--finetune checkpoint_path