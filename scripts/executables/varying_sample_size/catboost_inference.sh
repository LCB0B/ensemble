#!/bin/bash

for subset_size in 100 1000 10000 100000 1000000
do
    for split in val test
    do
        echo "Running inference for subset_size=$subset_size, split=$split"
        python tabular/catboost_inf.py \
            subset_size=$subset_size \
            data_split=$split \
            experiment_name=varying_size
    done
done
