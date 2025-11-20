trap 'kill 0' SIGINT

AUTO_NTOKENS=True
SCRIPT=scripts/destiny/finetune_nanoencoder_destiny-bg.py
NUM_WORKERS=30

DIR_PATH=destiny_dataset_bg
CHECKPOINT=destiny-pretrain/009_genuine_bison-pretrain-destiny_dataset_bg/best.ckpt


python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/EDU_High-school
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/EDU_Vocational
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/EDU_Higher

python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/DEM_Marriage
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/DEM_First-child
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/DEM_Migration

python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/SOC_Penal-crime
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/SOC_Drug-treatment
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/SOC_Preventative-measures

python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/LAB_Disability-pension
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/LAB_Millionare
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/LAB_Unemployment

python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Type2-diabetes
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Stroke
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Sleep-disorder
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_ADHD-medication
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Osteonecrosis
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Lung-cancer
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Antidepressants
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Colorectal-cancer
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes2010/HEA_Arrhythmia
