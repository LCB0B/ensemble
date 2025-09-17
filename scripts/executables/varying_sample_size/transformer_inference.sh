#!/bin/bash

for subset_size in 100 1000 10000 100000 1000000
do
    for pretrained in True False # Note capitalization for inference
    do
        run_id="pretrained_${pretrained}_size_${subset_size}"

        for split in val test
        do
            echo "Running inference for $run_id (split=$split)"

            python sequential/inf_nano.py \
                checkpoint_run_name=$run_id \
                checkpoint_experiment_name=varying_sample_size \
                inference_type=$split \
                output_file_name=$run_id \
                outcome_fname=at_risk
        done
    done
done
