#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

pretrain_experiment_name=pretrain_vulnerable_packing
finetune_experiment_name=finetune_vulnerable_packing
max_seq_len=2048
pretrain_run_id=pretrain
sample_folder=vulnerable_full
dir_path=vulnerable_full_lmdb
source_dir=vulnerable_full_prep
fast_dev_run=False
max_epoch_pretrain=2
max_epochs_at_risk=5
max_epochs_econ_vuln=10

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=at_risk \
  checkpoint_run_name=$pretrain_run_id \
  run_id=at_risk_pretrained_transformer \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_at_risk

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=econ_vuln \
  checkpoint_run_name=$pretrain_run_id \
  run_id=econ_vuln_pretrained_transformer \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_econ_vuln

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=at_risk \
  load_pretrained_model=False \
  run_id=at_risk_transformer \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_at_risk

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=econ_vuln \
  load_pretrained_model=False \
  run_id=econ_vuln_transformer \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_econ_vuln