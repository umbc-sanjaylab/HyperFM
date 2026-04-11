**CVPR Paper**: Zahid Hassan Tushar, and Sanjay Purushotham, "HyperFM: An Efficient Hyperspectral Foundation Model with Spectral Grouping", In proceedings of The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 (findings).

Please Cite the above paper if you use code or dataset.

*Dataset*: A subset of the HyperFM250k dataset is available on Zenodo at this link: https://zenodo.org/uploads/19495359



**HyperFM — Data download & preprocess**

This README explains how to use the repository scripts to download PACE data and run the preprocessing pipeline that produces paired HSI and target files for model training.

**Prerequisites**:
- **Python**: Python 3.8+ recommended.
- **Packages**: Install required Python packages. At minimum the downloader requires:
  `earthaccess`, `xarray`, `netCDF4`, `matplotlib`, `numpy`.
  Install with:

```bash
pip install earthaccess xarray netCDF4 matplotlib numpy
```
- **Earthdata credentials**: The downloader uses `earthaccess`. You must have a NASA Earthdata account and authenticate locally. The first run will prompt for credentials when using `earthaccess.login(persist=True)`.

**Overview**:
- **Script**: `data_download_and_preprocess.py` orchestrates the data download and preprocessing pipeline.
- **Downloader**: `util/data_download.py` contains `download_pace_oci_l1b` and `download_pace_l2_product` (Level-1B HSI and Level-2 CLD/CMK products).
- **Matching**: `util/sort_granules.py` is used to match HSI granules with target files and produces `matched_files.csv` and `unmatched_files.csv` in `raw_data/`.
- **Preprocessing**: `util/data_preprocess_w_GT.py` performs preprocessing and writes preprocessed HSI/target arrays to `preprocessed_data/` and produces `preprocessed_data/preprocessed_data_list.csv` describing the pairs.

**Quickstart (default run)**:
- Run the script with defaults (this will use the default date list and directories defined in the script):

```bash
python data_download_and_preprocess.py
```

This creates (if not present):
- `raw_data/hsi/` — downloaded Level-1B files
- `raw_data/target/` — downloaded Level-2 product files (CLD, CMK)
- `raw_data/matched_files.csv` — matches between HSI and target granules
- `raw_data/unmatched_files.csv` — unmatched granules
- `preprocessed_data/hsi/` and `preprocessed_data/target/` — preprocessed NumPy files
- `preprocessed_data/preprocessed_data_list.csv` — list of processed pairs

**Specifying dates and options**:
The script's argument parsing expects a Python list for `--date_list` (the file uses `type=list` for the argument).


This runs the pipeline for `2025-06-10` and downloads up to 2 granules per product.

**How the pipeline works (high level)**:
- For each date in `date_list` the code:
  - Downloads Level-1B HSI granules via `download_pace_oci_l1b` into `raw_data/hsi`.
  - Downloads Level-2 products (`CLD` and `CMK`) into `raw_data/target`.
  - Runs `util/sort_granules.find_matching_granule` to match HSI and target granules by timestamp and writes CSVs to `raw_data/`.
  - Runs `util/data_preprocess_w_GT.data_preprocess(...)` to create preprocessed arrays and a CSV index in `preprocessed_data/`.

**Common commands**:
- Authenticate to Earthdata (one-time; will prompt for credentials):

```bash
python -c "from earthaccess import login; login(persist=True)"
```

- Run with defaults:

```bash
python data_download_and_preprocess.py
```

- Run for a specific date programmatically 

**Tips & Troubleshooting**:
- If the script prints "No granules found", verify the date format and availability in the PACE datasets. Try a known date or a short date range.
- Ensure your machine can reach Earthdata (network & proxy settings). If behind a proxy, configure `HTTP(S)_PROXY` environment variables.
- If authentication fails, re-run the `earthaccess.login(persist=True)` command and follow prompts.
- If any package import fails, install the package with `pip install <package>`.

**Output structure (after a successful run)**:
- `raw_data/`
  - `hsi/` — downloaded Level-1B files (netCDF)
  - `target/` — downloaded Level-2 files
  - `matched_files.csv` — matched pairs
  - `unmatched_files.csv` — unmatched granules
- `preprocessed_data/`
  - `hsi/` — preprocessed HSI `.npy` files
  - `target/` — preprocessed target `.npy` files
  - `preprocessed_data_list.csv` — CSV listing paired preprocessed files for training

**Caution**:
- If you plan to download large volumes, increase `--max_granules` carefully and ensure you have sufficient disk space.

**Pretraining (`main_pretrain.py`)**

- **Overview**: Use `main_pretrain.py` to run masked autoencoder pretraining. The repository includes `pretrain.sh` as a runnable example that checks GPUs, logs output, sets `CUDA_VISIBLE_DEVICES`, and launches training with `torchrun` for multi-GPU runs.

- **What `pretrain.sh` does**:
  - Creates a timestamped log file under `output_dir/` and redirects both stdout and stderr into it while still printing to the console (`tee`).
  - Runs `nvidia-smi` checks to show GPU status and memory.
  - Exports `CUDA_VISIBLE_DEVICES` to restrict visible GPUs.
  - Calls `torchrun` with `--nproc-per-node` equal to number of processes (GPUs) and a random rendezvous port (`rdzv_endpoint=localhost:$(shuf -i 20000-30000 -n 1)`) to start distributed training.

- **Example command (from the current `pretrain.sh`)**:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc-per-node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:$(shuf -i 20000-30000 -n 1) \
  main_pretrain.py \
  --batch_size 16 --accum_iter 16 \
  --epochs 250 --warmup_epochs 20 --patience 30 \
  --input_size 96 --patch_size 8 \
  --mask_ratio 0.75 \
  --model_type hyperFM --model hyperFM_enc4 --tt_rank 3 \
  --blr 0.001 --num_workers 6 \
  --data_root_dir preprocessed_data \
  --train_path preprocessed_data/preprocessed_data_list.csv \
  --val_path preprocessed_data/preprocessed_data_list.csv \
  --test_path preprocessed_data/preprocessed_data_list.csv \
  --output_dir ./output_dir \
  --log_dir ./output_dir
```

- **Key CLI arguments**:
  - `--batch_size`: per-GPU batch size.
  - `--accum_iter`: gradient accumulation steps to increase effective batch size.
  - `--epochs`, `--warmup_epochs`, `--patience`: control training length and early stopping.
  - `--input_size`, `--patch_size`: model input and patch dimensions.
  - `--mask_ratio`: fraction of patches masked for MAE-style pretraining.
  - `--model_type`, `--model`, `--tt_rank`: model selection and factorization rank.
  - `--blr`/`--lr`: base learning rate; if `--lr` is not provided the script computes it from `--blr` and effective batch size.
  - `--num_workers`: DataLoader workers.
  - `--data_root_dir`, `--train_path`, `--val_path`, `--test_path`: point to the `preprocessed_data` produced by preprocessing.
  - `--output_dir`, `--log_dir`: paths for saving checkpoints and tensorboard logs. Note: `main_pretrain.py` appends a timestamped `./outputs/<timestamp>` subfolder to `--output_dir` when creating directories.

- **Practical notes**:
  - Adjust `CUDA_VISIBLE_DEVICES` to the GPU indices available on your machine.
  - For single-GPU testing use `--nproc-per-node=1` and set `CUDA_VISIBLE_DEVICES` accordingly.
  - Ensure `preprocessed_data/preprocessed_data_list.csv` exists and references valid `.npy` files created by the preprocessing step.
  - The log file path is controlled by `pretrain.sh` (`LOGFILE`); check it when debugging startup or distributed rendezvous issues.
  - Reduce `--epochs`, `--num_workers`, and `--accum_iter` for quick smoke tests.

**Finetuning (`main_finetune.py`)**

- **Overview**: Use `main_finetune.py` to finetune the pre-trained encoder + decoder (or decoder-only) for downstream regression tasks. The repository includes `finetune.sh` as a runnable example that performs GPU checks, logs output, sets `CUDA_VISIBLE_DEVICES`, and launches multi-GPU finetuning with `torchrun`.

- **What `finetune.sh` does**:
  - Creates a timestamped log file under `finetune_dir/` and redirects both stdout and stderr into it while still printing to the console (`tee`).
  - Runs `nvidia-smi` checks to show GPU status and memory.
  - Exports `CUDA_VISIBLE_DEVICES` to restrict visible GPUs.
  - Calls `torchrun` with `--nproc-per-node` equal to number of processes (GPUs) and a random rendezvous port (`rdzv_endpoint=localhost:$(shuf -i 20000-30000 -n 1)`) to start distributed training.

- **Example command (from the current `finetune.sh`)**:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc-per-node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:$(shuf -i 20000-30000 -n 1) \
  main_finetune.py \
  --batch_size 16 --accum_iter 16 \
  --epochs 250 --warmup_epochs 20 --patience 30 \
  --input_size 96 --patch_size 8 \
  --model_type hyperFM_conv --model hyperFM_enc4 --nb_classes 4 \
  --dataset_type pace --data_mode all \
  --blr 0.001 --num_workers 6 \
  --train_path preprocessed_data/preprocessed_data_list.csv \
  --val_path preprocessed_data/preprocessed_data_list.csv \
  --test_path preprocessed_data/preprocessed_data_list.csv \
  --output_dir ./finetune_dir \
  --log_dir ./finetune_dir \
  --finetune_dec_only \
  --finetune checkpoint_path
```

- **Key CLI arguments**:
  - `--batch_size`: per-GPU batch size.
  - `--accum_iter`: gradient accumulation steps to increase effective batch size.
  - `--epochs`, `--warmup_epochs`, `--patience`: control training length and early stopping.
  - `--input_size`, `--patch_size`: model input and patch dimensions.
  - `--model_type`, `--model`, `--tt_rank`: model selection and factorization rank for the architecture.
  - `--nb_classes`: number of output channels / tasks for the decoder (e.g., 4 for COT/CER/CWP/CTH).
  - `--data_mode`: which target(s) to train on (`cot`, `cer`, `cth`, `cwp`, `cmask`, or `all`).
  - `--finetune_dec_only`: freeze encoder and train decoder only.
  - `--finetune`: path to the checkpoint to load (the script loads checkpoint and applies it with `strict=False`).
  - `--eval` and `--eval_checkpoint`: run evaluation-only using a specified checkpoint.
  - `--blr`/`--lr`, `--layer_decay`, `--weight_decay`: optimizer hyperparameters; `--lr` is computed from `--blr` and effective batch size if not provided.
  - `--num_workers`, `--pin_mem`: DataLoader performance options.
  - `--output_dir`, `--log_dir`: paths for saving checkpoints and TensorBoard logs. Note: `main_finetune.py` appends a timestamped `./outputs/<timestamp>` subfolder to `--output_dir` when creating directories.

- **Practical notes**:
  - Set `CUDA_VISIBLE_DEVICES` to the GPU indices available on your machine.
  - For single-GPU testing use `--nproc-per-node=1` and set `CUDA_VISIBLE_DEVICES` accordingly.
  - Provide a valid `--finetune` checkpoint (path) when resuming or finetuning from pretrained weights.
  - Use `--finetune_dec_only` to quickly fine-tune a small decoder head if you want fast adaptation with limited compute.
  - When `--eval` is used the script will print per-target MSEs (for `data_mode=all`) and exit.
  - Reduce `--epochs`, `--num_workers`, and `--accum_iter` for quick smoke tests; check the generated `finetune_dir` logs for diagnostics.


