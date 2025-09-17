#!/usr/bin/env bash
# Exit immediately if any command fails
set -e
gpa_experiment_name=socmob_full_gpa_tabular
income_experiment_name=socmob_full_income_tabular
data_folder=socmob_full_joined
n_trials=50
max_epochs=10
batch_size_lr=4096 #32768
batch_size_mlp=4096 #32768
num_workers=8
for censor in 0 5 10 15; do
  for group in both own parents; do
    echo "-------------------------------------------------------------------------"
    echo "Outcome: GPA" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "Logistic Regression"
    echo "-------------------------------------------------------------------------"
    python tabular/lr_train.py \
      experiment_name=$gpa_experiment_name \
      n_trials=$n_trials\
      data_folder=$data_folder\
      outcome=gpa_censor_${censor}_${group}\
      max_epochs=$max_epochs\
      batch_size=$batch_size_lr\
      num_workers=$num_workers

      
    echo "-------------------------------------------------------------------------"
    echo "Outcome: GPA" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "MLP"
    echo "-------------------------------------------------------------------------"
    python tabular/mlp_train.py \
      experiment_name=$gpa_experiment_name \
      n_trials=$n_trials\
      data_folder=$data_folder\
      outcome=gpa_censor_${censor}_${group}\
      max_epochs=$max_epochs\
      batch_size=$batch_size_mlp\
      num_workers=$num_workers
  done
done

for censor in 0 5 10 15 20 25 30; do
  for group in both own parents; do

  
    echo "-------------------------------------------------------------------------"
    echo "Outcome: Income" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "Logistic Regression"
    echo "-------------------------------------------------------------------------"
    python tabular/lr_train.py \
      experiment_name=$income_experiment_name \
      n_trials=$n_trials\
      data_folder=$data_folder\
      outcome=income_censor_${censor}_${group}\
      max_epochs=$max_epochs\
      batch_size=$batch_size_lr\
      num_workers=$num_workers

    echo "-------------------------------------------------------------------------"
    echo "Outcome: Income" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "MLP"
    echo "-------------------------------------------------------------------------"
    python tabular/mlp_train.py \
      experiment_name=$income_experiment_name \
      n_trials=$n_trials\
      data_folder=$data_folder\
      outcome=income_censor_${censor}_${group}\
      max_epochs=$max_epochs\
      batch_size=$batch_size_mlp\
      num_workers=$num_workers
  done
done
