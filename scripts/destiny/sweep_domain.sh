trap 'kill 0' SIGINT

SCRIPT=scripts/destiny/finetune_nanoencoder_destiny.py
NUM_WORKERS=15
AUTO_NTOKENS=True

DIR_PATH=destiny_dataset_edu
CHECKPOINT=destiny-pretrain/012_prudent_whale-pretrain-destiny_dataset_edu/best.ckpt
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/EDU_High-school
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/EDU_Vocational
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/EDU_Higher

DIR_PATH=destiny_dataset_demo
CHECKPOINT=destiny-pretrain/011_healthy_otter-pretrain-destiny_dataset_demo/best.ckpt
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/DEM_Marriage
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/DEM_First-child
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/DEM_Migration

DIR_PATH=destiny_dataset_soc
CHECKPOINT=destiny-pretrain/013_zany_sphinx-pretrain-destiny_dataset_soc/best.ckpt
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/SOC_Penal-crime
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/SOC_Drug-treatment
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/SOC_Preventative-measures

DIR_PATH=destiny_dataset_labour
CHECKPOINT=destiny-pretrain/014_considerate_bear-pretrain-destiny_dataset_labour/best.ckpt
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/LAB_Disability-pension
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/LAB_Millionare
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/LAB_Unemployment

DIR_PATH=destiny_dataset_health
CHECKPOINT=destiny-pretrain/017_itchy_chameleon-pretrain-destiny_dataset_health/best.ckpt
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Type2-diabetes
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Stroke
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Sleep-disorder
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_ADHD-medication
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Osteonecrosis
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Lung-cancer
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Antidepressants
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Colorectal-cancer
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Arrhythmia
