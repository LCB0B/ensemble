#!/bin/bash
# Exit immediately if any command fails
set -e

for subset_size in 100 1000 10000 100000 1000000
do
    for pretrained in true false
    do
        run_id="pretrained_${pretrained}_subset_size_${subset_size}"

        echo "Training with run_id=$run_id"

        python sequential/fine_nano.py \
            subset_size=$subset_size \
            experiment_name=varying_sample_size_final\
            load_pretrained_model=$pretrained \
            checkpoint_experiment_name=pretrain_vulnerable_final \
            run_id=$run_id\
            outcome_fname=at_risk
    done
done
