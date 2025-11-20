trap 'kill 0' SIGINT

SCRIPT=scripts/destiny/finetune_nanoencoder_destiny2.py
NUM_WORKERS=10
AUTO_NTOKENS=True

# DIR_PATH=destiny_dataset_noedu
# CHECKPOINT=destiny-pretrain/022_cooperative_chameleon-pretrain-destiny_dataset_noedu/best.ckpt
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/EDU_High-school
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/EDU_Vocational
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/EDU_Higher

DIR_PATH=destiny_dataset_nodemo
CHECKPOINT=destiny-pretrain/023_colorful_centaur-pretrain-destiny_dataset_nodemo/best.ckpt
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/DEM_Marriage
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/DEM_First-child
python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/DEM_Migration

# DIR_PATH=destiny_dataset_nosoc
# CHECKPOINT=destiny-pretrain/021_exotic_bison-pretrain-destiny_dataset_nosoc/best.ckpt
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/SOC_Penal-crime
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/SOC_Drug-treatment
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/SOC_Preventative-measures

# DIR_PATH=destiny_dataset_nolabour
# CHECKPOINT=destiny-pretrain/019_clever_dragon-pretrain-destiny_dataset_nolabour/best.ckpt
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/LAB_Disability-pension
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/LAB_Disability-pension2
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/LAB_Millionare
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/LAB_Unemployment

# DIR_PATH=destiny_dataset_nohealth
# CHECKPOINT=destiny-pretrain/018_stormy_stallion-pretrain-destiny_dataset_nohealth/best.ckpt
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_type2-diabetes
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_stroke
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_sleep-disorder
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_schizophrenia
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_osteonecrosis
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_lung-cancer
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_depression
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_colorectal-cancer
# python "$SCRIPT" dir_path="$DIR_PATH" load_pretrained_model="$CHECKPOINT" num_workers="$NUM_WORKERS" auto_ntokens="$AUTO_NTOKENS" outcome=outcomes/HEA_arrhythmia
