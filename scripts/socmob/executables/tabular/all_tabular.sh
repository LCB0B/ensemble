#!/usr/bin/env bash
# Exit immediately if any command fails
set -e
inference_data_folder=vulnerable_full_extended_preds

# echo "LR train"
# python tabular/lr_train.py -m experiment_name=lr_extended outcome=econ_vuln,at_risk
echo "LR inference"
python tabular/lr_inference.py -m experiment_name=lr_extended outcome=econ_vuln,at_risk data_split=test,val output_folder=$inference_data_folder
# echo "CB train"
# python tabular/catboost_train.py -m experiment_name=cb_extended outcome=econ_vuln,at_risk
echo "CB inference"
python tabular/catboost_inf.py -m experiment_name=cb_extended outcome=econ_vuln,at_risk data_split=test,val output_folder=$inference_data_folder