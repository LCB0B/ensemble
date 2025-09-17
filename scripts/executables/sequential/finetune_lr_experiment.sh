#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

pretrain_experiment_name=pretrain_vulnerable_final
finetune_experiment_name=finetune_vulnerable_final
max_seq_len=8192
pretrain_run_id=pretrain
sample_folder=vulnerable_full
dir_path=vulnerable_full_final_lmdb
source_dir=vulnerable_full_extended_prep
fast_dev_run=False
max_epoch_pretrain=5
max_epochs_at_risk=5
max_epochs_econ_vuln=10
warmup_steps=1000


for learning_rate in 5e-6
do
  # run_id="at_risk_pretrained_lr_${learning_rate}"
  # python sequential/fine_nano.py \
  #   fast_dev_run=$fast_dev_run \
  #   sample_folder=$sample_folder \
  #   source_dir=$source_dir \
  #   outcome_fname=at_risk \
  #   checkpoint_run_name=$pretrain_run_id \
  #   run_id=$run_id \
  #   experiment_name=$finetune_experiment_name \
  #   checkpoint_experiment_name=$pretrain_experiment_name \
  #   dir_path=$dir_path \
  #   max_seq_len=$max_seq_len \
  #   max_epochs=$max_epochs_at_risk\
  #   learning_rate=$learning_rate\
  #   lengths=lengths_at_risk_targets

  run_id="at_risk_not_pretrained_lr_${learning_rate}"
  python sequential/fine_nano.py \
    fast_dev_run=$fast_dev_run \
    sample_folder=$sample_folder \
    source_dir=$source_dir \
    outcome_fname=at_risk \
    load_pretrained_model=False \
    run_id=$run_id \
    experiment_name=$finetune_experiment_name \
    checkpoint_experiment_name=$pretrain_experiment_name \
    dir_path=$dir_path \
    max_seq_len=$max_seq_len \
    max_epochs=$max_epochs_at_risk\
    learning_rate=$learning_rate\
    lengths=lengths_at_risk_targets

done

# for learning_rate in 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
# do
#   run_id="at_risk_pretrained_lr_${learning_rate}"
#   python sequential/fine_nano.py \
#     fast_dev_run=$fast_dev_run \
#     sample_folder=$sample_folder \
#     source_dir=$source_dir \
#     outcome_fname=at_risk \
#     checkpoint_run_name=$pretrain_run_id \
#     run_id=$run_id \
#     experiment_name=$finetune_experiment_name \
#     checkpoint_experiment_name=$pretrain_experiment_name \
#     dir_path=$dir_path \
#     max_seq_len=$max_seq_len \
#     max_epochs=$max_epochs_at_risk\
#     learning_rate=$learning_rate\
#     lengths=lengths_at_risk_targets

#   run_id="at_risk_not_pretrained_lr_${learning_rate}"
#   python sequential/fine_nano.py \
#     fast_dev_run=$fast_dev_run \
#     sample_folder=$sample_folder \
#     source_dir=$source_dir \
#     outcome_fname=at_risk \
#     load_pretrained_model=False \
#     run_id=$run_id \
#     experiment_name=$finetune_experiment_name \
#     checkpoint_experiment_name=$pretrain_experiment_name \
#     dir_path=$dir_path \
#     max_seq_len=$max_seq_len \
#     max_epochs=$max_epochs_at_risk\
#     learning_rate=$learning_rate\
#     lengths=lengths_at_risk_targets

# done

## ECON VULN LR FINETUNE
# Intention
# for learning_rate in 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
# do
#  run_id="at_risk_pretrained_lr_${learning_rate}_warmup_steps_${warmup_steps}"
#  python sequential/fine_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   outcome_fname=econ_vuln \
#   checkpoint_run_name=$pretrain_run_id \
#   run_id=$run_id \
#   experiment_name=$finetune_experiment_name \
#   checkpoint_experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epochs_econ_vuln\
#   learning_rate=$learning_rate\
#   warmup_steps=$warmup_steps\
#   lengths=lengths_econ_vuln_targets
# done


# learning_rate=1e-5
# run_id="econ_vuln_not_pretrained_lr_${learning_rate}_warmup_steps_${warmup_steps}"
# python sequential/fine_nano.py \
#   fast_dev_run=$fast_dev_run \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   outcome_fname=econ_vuln \
#   load_pretrained_model=False \
#   checkpoint_run_name=$pretrain_run_id \
#   run_id=$run_id \
#   experiment_name=$finetune_experiment_name \
#   checkpoint_experiment_name=$pretrain_experiment_name \
#   dir_path=$dir_path \
#   max_seq_len=$max_seq_len \
#   max_epochs=$max_epochs_econ_vuln\
#   learning_rate=$learning_rate\
#   warmup_steps=$warmup_steps\
#   lengths=lengths_econ_vuln_targets