pretrain_experiment_name=pretrain_vulnerable
finetune_experiment_name=finetune_vulnerable
pretrain_run_id=pretrain
sample_folder=vulnerable_full
dir_path=vulnerable_full_lmdb
source_dir=vulnerable_full_prep
fast_dev_run=False
python sequential/fine_nano.py \
  fast_dev_run=$fast_dev_run \
  sample_folder=$sample_folder \
  source_dir=$source_dir \
  outcome_fname=at_risk \
  load_pretrained_model=False \
  run_id=at_risk_transformer \
  experiment_name=$finetune_experiment_name \
  checkpoint_experiment_name=$pretrain_experiment_name \
  dir_path=$dir_path