#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES="0"  # Use GPU 0
export POLARS_MAX_THREADS=8      # Limit Polars threads
export RAYON_NUM_THREADS=8       # Limit Rayon threads
export OMP_NUM_THREADS=8         # Limit OpenMP threads
export OMP_THREADS_LIMIT=8       # Another OpenMP thread limit


echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"


# Run Python scripts sequentially
#python scripts/pretrain_nanoencoder2.py
python scripts/finetune_nanoencoder_risk.py
python scripts/inference_nanoencoder.py
