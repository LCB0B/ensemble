#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

pretrain_experiment_name=pretrain_vulnerable_bidir_experiment
finetune_experiment_name=finetune_vulnerable_bidir_experiment
pretrain_run_id=pretrain
sample_folder=vulnerable_full
dir_path=vulnerable_full_extended_lmdb
source_dir=vulnerable_full_extended_prep
fast_dev_run=False
declare -i max_seq_len=2048
declare -i max_epoch_pretrain=2


python sequential/pre_nano_fail_log.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  run_id=$pretrain_run_id \
  experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epoch_pretrain \
  val_check_interval=2500 
