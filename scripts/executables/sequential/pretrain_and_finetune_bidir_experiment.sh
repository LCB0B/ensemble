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
declare -i max_epochs_at_risk=5
declare -i max_epochs_econ_vuln=10


# python sequential/pre_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   run_id=$pretrain_run_id \
#   experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epoch_pretrain

# python sequential/fine_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   outcome_fname=at_risk \
#   checkpoint_run_name=$pretrain_run_id \
#   run_id=at_risk_pretrained_transformer \
#   experiment_name=$finetune_experiment_name \
#   checkpoint_experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epochs_at_risk\
#   lengths=lengths_at_risk_targets

# python sequential/fine_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   outcome_fname=econ_vuln \
#   checkpoint_run_name=$pretrain_run_id \
#   run_id=econ_vuln_pretrained_transformer \
#   experiment_name=$finetune_experiment_name \
#   checkpoint_experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epochs_econ_vuln\
#   lengths=lengths_econ_vuln_targets

# python sequential/fine_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   outcome_fname=at_risk \
#   load_pretrained_model=False \
#   run_id=at_risk_transformer \
#   experiment_name=$finetune_experiment_name \
#   checkpoint_experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epochs_at_risk\
#   lengths=lengths_at_risk_targets

# python sequential/fine_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   outcome_fname=econ_vuln \
#   load_pretrained_model=False \
#   run_id=econ_vuln_transformer \
#   experiment_name=$finetune_experiment_name \
#   checkpoint_experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epochs_econ_vuln \
#   lengths=lengths_econ_vuln_targets

# Bidirectional

python sequential/pre_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  run_id=pretrain_bidir \
  experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epoch_pretrain \
  causal=False

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=at_risk \
  checkpoint_run_name=$pretrain_run_id \
  run_id=at_risk_pretrained_transformer_bidir \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_at_risk\
  lengths=lengths_at_risk_targets \
  causal=False

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=econ_vuln \
  checkpoint_run_name=$pretrain_run_id \
  run_id=econ_vuln_pretrained_transformer_bidir \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_econ_vuln\
  lengths=lengths_econ_vuln_targets \
  causal=False

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=at_risk \
  load_pretrained_model=False \
  run_id=at_risk_transformer_bidir \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_at_risk\
  lengths=lengths_at_risk_targets \
  causal=False

python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=econ_vuln \
  load_pretrained_model=False \
  run_id=econ_vuln_transformer_bidir \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path \
  max_seq_len=$max_seq_len \
  max_epochs=$max_epochs_econ_vuln \
  lengths=lengths_econ_vuln_targets \
  causal=False