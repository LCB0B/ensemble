#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

checkpoint_experiment_name=finetune_vulnerable_bidir_experiment
pretrain_run_id=pretrain
sample_folder=vulnerable_full
dir_path=vulnerable_full_extended_lmdb
source_dir=vulnerable_full_extended_prep
inference_data_folder=vulnerable_full_extended_preds

python sequential/inf_nano.py -m \
  outcome_fname=at_risk \
  output_file_name=pretrained \
  inference_type=test,val \
  checkpoint_run_name=at_risk_pretrained_transformer \
  lengths=lengths_at_risk_targets \
  checkpoint_experiment_name=$checkpoint_experiment_name \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  dir_path=$dir_path \
  inference_data_folder=$inference_data_folder

python sequential/inf_nano.py -m \
  outcome_fname=econ_vuln \
  output_file_name=pretrained \
  inference_type=test,val \
  checkpoint_run_name=econ_vuln_pretrained_transformer \
  lengths=lengths_econ_vuln_targets \
  checkpoint_experiment_name=$checkpoint_experiment_name \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  dir_path=$dir_path \
  inference_data_folder=$inference_data_folder

python sequential/inf_nano.py -m \
  outcome_fname=at_risk \
  output_file_name=finetuned \
  inference_type=test,val \
  checkpoint_run_name=at_risk_transformer \
  lengths=lengths_at_risk_targets \
  checkpoint_experiment_name=$checkpoint_experiment_name\
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  dir_path=$dir_path \
  inference_data_folder=$inference_data_folder

python sequential/inf_nano.py -m \
  outcome_fname=econ_vuln \
  output_file_name=finetuned \
  inference_type=test,val \
  checkpoint_run_name=econ_vuln_transformer \
  lengths=lengths_econ_vuln_targets \
  checkpoint_experiment_name=$checkpoint_experiment_name\
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  dir_path=$dir_path \
  inference_data_folder=$inference_data_folder