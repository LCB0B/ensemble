#!/bin/bash

for subset_size in 100 1000 10000 100000 1000000
do
    echo "Running training for subset_size=$subset_size"
    python tabular/lr_train.py subset_size=$subset_size experiment_name=varying_size
done
