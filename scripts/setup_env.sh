#!/bin/bash
# Use all 4 GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# Only use 8 polars thread on the CPU
# export POLARS_MAX_THREADS=8
# export RAYON_NUM_THREADS=8
# export OMP_NUM_THREADS=8
# export OMP_THREADS_LIMIT=8