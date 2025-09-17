#!/usr/bin/env bash
set -e

# Execute from K:\project2vec\project2vec\scripts\socmob
# bash executables/sequential/full_pipeline.sh

# Global shared config
sample_folder="socmob_full_sample"
dir_path="socmob_full_sample_lmdb"
source_dir="socmob_full_sample_prep"
max_seq_len=8192
fast_dev_run=False
max_epochs_pretrain=1
max_epochs_finetune=1
inference_data_folder="socmob_full_sample_preds" 

if [ "$fast_dev_run" = "True" ]; then
  echo "Pretrained model not loaded due to fast_dev_run = True"
  load_pretrained_model=False
else
  load_pretrained_model=True
fi
# Constant experiment names
pretrain_experiment_name="socmob_full_pretrain"
finetune_experiment_name="socmob_full_finetune"

# Feature set configs
declare -A FEATURE_SETS
FEATURE_SETS["childmotherfather"]="Child Mother Father"
FEATURE_SETS["motherfather"]="Mother Father"
FEATURE_SETS["child"]="Child"


# FULL FAMILY FIRST! This is to ensure everyone is written to lmdb
#ORDERED_KEYS=("childmotherfather" "motherfather" "child")
ORDERED_KEYS=("childmotherfather")
# key="childmotherfather"
# deterministic_run_id_pretrain="pretrain_${key}_continued"
# deterministic_run_id_pretrain_checkpoint="pretrain_${key}"
# feature_set_str="[$(echo $key | sed 's/ /, /g')]"
# echo ">>> Continuing pretraining: ${key} <<<"

# python sequential/pretrain_nanoencoder_family.py \
#   experiment_name=$pretrain_experiment_name \
#   deterministic_run_id=$deterministic_run_id_pretrain \
#   sample_folder=$sample_folder \
#   source_dir=$source_dir \
#   dir_path=$dir_path \
#   feature_set="$feature_set_str" \
#   max_seq_len=$max_seq_len \
#   fast_dev_run=$fast_dev_run \
#   max_epochs=$max_epochs_pretrain \
#   load_failed_model=True \
#   failed_experiment_name=$pretrain_experiment_name \
#   failed_run_name=$deterministic_run_id_pretrain_checkpoint \
#   num_workers=16 \
#   n_tokens=250000

# ---------- CONTINUE PRETRAIN ----------
for key in "${ORDERED_KEYS[@]}"; do
  fs="${FEATURE_SETS[$key]}"
  feature_set_str="[$(echo $fs | sed 's/ /, /g')]"

  
  deterministic_run_id_pretrain="pretrain_${key}_continued"
  deterministic_run_id_pretrain_checkpoint="pretrain_${key}"

  echo ">>> Pretraining: ${fs} <<<"
  python sequential/pretrain_nanoencoder_family.py \
    experiment_name=$pretrain_experiment_name \
    deterministic_run_id=$deterministic_run_id_pretrain \
    sample_folder=$sample_folder \
    source_dir=$source_dir \
    dir_path=$dir_path \
    feature_set="$feature_set_str" \
    max_seq_len=$max_seq_len \
    fast_dev_run=$fast_dev_run \
    max_epochs=$max_epochs_pretrain\
    load_failed_model=True \
    failed_experiment_name=$pretrain_experiment_name \
    failed_run_name=$deterministic_run_id_pretrain_checkpoint \
    num_workers=16 \
    n_tokens=250000

done 

# # ---------- PRETRAIN ----------
# for key in "${ORDERED_KEYS[@]}"; do
#   fs="${FEATURE_SETS[$key]}"
#   feature_set_str="[$(echo $fs | sed 's/ /, /g')]"

#   deterministic_run_id_pretrain="pretrain_${key}"

#   echo ">>> Pretraining: ${fs} <<<"
#   python sequential/pretrain_nanoencoder_family.py \
#     experiment_name=$pretrain_experiment_name \
#     deterministic_run_id=$deterministic_run_id_pretrain \
#     sample_folder=$sample_folder \
#     source_dir=$source_dir \
#     dir_path=$dir_path \
#     feature_set="$feature_set_str" \
#     max_seq_len=$max_seq_len \
#     fast_dev_run=$fast_dev_run \
#     max_epochs=$max_epochs_pretrain
# done 

# # ---------- FINETUNE ----------
# for key in "${ORDERED_KEYS[@]}"; do
#   fs="${FEATURE_SETS[$key]}"
#   feature_set_str="[$(echo $fs | sed 's/ /, /g')]"

#   deterministic_run_id_pretrain="pretrain_${key}"
#   for outcome in income gpa; do
#     deterministic_run_id_finetune="${outcome}_${key}"

#     echo ">>> Finetuning: ${fs} for ${outcome} <<<"
#     python sequential/finetune_nanoencoder_family.py \
#       experiment_name=$finetune_experiment_name \
#       deterministic_run_id=$deterministic_run_id_finetune \
#       outcome=$outcome \
#       sample_folder=$sample_folder \
#       source_dir=$source_dir \
#       dir_path=$dir_path \
#       feature_set="$feature_set_str" \
#       max_seq_len=$max_seq_len \
#       fast_dev_run=$fast_dev_run \ \
#       checkpoint_run_name=$deterministic_run_id_pretrain \
#       checkpoint_experiment_name=$pretrain_experiment_name \
#       load_pretrained_model=$load_pretrained_model \
#       max_epochs=$max_epochs_finetune
#   done
# done


# # -------- Inference for all (val + test) --------

# if [ "$fast_dev_run" != "True" ]; then
#     for key in "${ORDERED_KEYS[@]}"; do
#     fs="${FEATURE_SETS[$key]}"
#     feature_set_str="[$(echo $fs | sed 's/ /, /g')]"

#     for outcome in income gpa; do
#         checkpoint_run_id="${outcome}_${key}"

#         for inference_type in val test; do
#         echo ">>> Inference: ${fs} | ${outcome} | ${inference_type} <<<"
#         python sequential/inference_nanoencoder_family.py \
#             experiment_name=$finetune_experiment_name \
#             sample_folder=$sample_folder \
#             source_dir=$source_dir \
#             dir_path=$dir_path \
#             feature_set="$feature_set_str" \
#             outcome=$outcome \
#             checkpoint_experiment_name=$finetune_experiment_name \
#             checkpoint_run_name=$checkpoint_run_id \
#             inference_type=$inference_type \
#             inference_data_folder=$inference_data_folder \
#             max_seq_len=$max_seq_len
#         done
#     done
#     done
# else
#   echo "Skipping inference because fast_dev_run=True"
# fi