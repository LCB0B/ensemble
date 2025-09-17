#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

pretrain_experiment_name=vulnerable_pretrain_experiments
finetune_experiment_name=vulnerable_finetune_experiments
sample_folder=vulnerable_full
dir_path=vulnerable_full_extended_lmdb
source_dir=vulnerable_full_extended_prep
fast_dev_run=False
n_tokens=550000
declare -i max_seq_len=8192
declare -i max_epoch_pretrain=5

# Intention
#for learning_rate in 1e-5 5e-5 1e-4 5e-4 1e-3
#do
#  run_id="lr_${learning_rate}"
#  python sequential/pre_nano.py\
#    fast_dev_run=$fast_dev_run \
#    sample_folder=$sample_folder \
#    source_dir=$source_dir \
#    run_id=$run_id \
#    experiment_name=$pretrain_experiment_name \
#    dir_path=$dir_path \
#    max_seq_len=$max_seq_len \
#    max_epochs=$max_epoch_pretrain\
#    n_tokens=$n_tokens\
#    learning_rate=$learning_rate
#done

# First success

# python sequential/pre_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   run_id="lr_5e-5_restart" \
#   experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epoch_pretrain \
#   n_tokens=$n_tokens \
#   learning_rate=5e-5 \
#   load_failed_model=true \
#   failed_experiment_name=vulnerable_pretrain_experiments \
#   failed_run_name=lr_5e-5


# python sequential/pre_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   run_id="lr_1e-4" \
#   experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epoch_pretrain \
#   n_tokens=$n_tokens \
#   learning_rate=1e-4

python sequential/pre_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  run_id="lr_1e-4_restart" \
  experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epoch_pretrain \
  n_tokens=$n_tokens \
  learning_rate=1e-4 \
  load_failed_model=true \
  failed_experiment_name=vulnerable_pretrain_experiments \
  failed_run_name=lr_1e-4

python sequential/pre_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  run_id="lr_5e-4" \
  experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epoch_pretrain \
  n_tokens=$n_tokens \
  learning_rate=5e-4

python sequential/pre_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  run_id="lr_1e-3" \
  experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epoch_pretrain \
  n_tokens=$n_tokens \
  learning_rate=1e-3