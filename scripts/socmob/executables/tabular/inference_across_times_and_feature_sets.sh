#!/usr/bin/env bash
# Exit immediately if any command fails
set -e
gpa_experiment_name=socmob_full_gpa_tabular
income_experiment_name=socmob_full_income_tabular
output_folder=socmob_full_preds_tabular
data_folder=socmob_full_joined
batch_size=64
num_workers=8

for censor in 0 5 10 15; do
  for group in both own parents; do
    echo "-------------------------------------------------------------------------"
    echo "Outcome: GPA" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "Logistic Regression"
    echo "-------------------------------------------------------------------------"
    python tabular/lr_inference.py -m \
      experiment_name=$gpa_experiment_name \
      data_folder=$data_folder\
      outcome=gpa_censor_${censor}_${group}\
      batch_size=$batch_size\
      num_workers=$num_workers\
      output_folder=$output_folder\
      data_split=test,val

      
    echo "-------------------------------------------------------------------------"
    echo "Outcome: GPA" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "MLP"
    echo "-------------------------------------------------------------------------"
    python tabular/mlp_inference.py -m \
      experiment_name=$gpa_experiment_name \
      data_folder=$data_folder\
      outcome=gpa_censor_${censor}_${group}\
      batch_size=$batch_size\
      num_workers=$num_workers\
      output_folder=$output_folder\
      data_split=test,val
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
    python tabular/lr_inference.py -m \
      experiment_name=$income_experiment_name \
      data_folder=$data_folder\
      outcome=income_censor_${censor}_${group}\
      batch_size=$batch_size\
      num_workers=$num_workers\
      output_folder=$output_folder\
      data_split=test,val

  
    echo "-------------------------------------------------------------------------"
    echo "Outcome: Income" 
    echo "Censor years: ${censor}" 
    echo "Feature set: ${group}" 
    echo "MLP"
    echo "-------------------------------------------------------------------------"
    python tabular/mlp_inference.py -m \
      experiment_name=$income_experiment_name \
      data_folder=$data_folder\
      outcome=income_censor_${censor}_${group}\
      batch_size=$batch_size\
      num_workers=$num_workers\
      output_folder=$output_folder\
      data_split=test,val
  done
done
