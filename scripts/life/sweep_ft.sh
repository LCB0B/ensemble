trap 'kill 0' SIGINT

python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/depression 
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/schizophrenia
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/type2-diabetes 
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/osteonecrosis
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/colorectal-cancer
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/lung-cancer
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/arrhythmia
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/stroke
python scripts/destiny/finetune_nanoencoder_destiny.py dir_path=destiny_dataset_health n_tokens=700000 outcome.train=outcomes/sleep-disorder