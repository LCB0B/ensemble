#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

pretrain_experiment_name=pretrain_vulnerable_seqlen_2k
finetune_experiment_name=finetune_vulnerable_seqlen_2k
max_seq_len=2048
pretrain_run_id=test_packing_no_padding
sample_folder=vulnerable_full
dir_path=vulnerable_full_lmdb
source_dir=vulnerable_full_prep
fast_dev_run=False
learning_rate=1e-4

python sequential/pre_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  run_id=$pretrain_run_id \
  experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  learning_rate=$learning_rate
