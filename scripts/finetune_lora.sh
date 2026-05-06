#!/bin/bash
export PYTHONPATH="/data_lyn/Neo_Red:$PYTHONPATH"
# Set the following variables correspondingly to run this script:
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

##################禁用flash_atention#############
export USE_FLASH_ATTENTION=0
export XFORMERS_DISABLE=1
export FLASH_ATTENTION_SKIP_CUSTOM=1
export TRANSFORMERS_USE_FLASH_ATTENTION=0

################## VICUNA ##################
PROMPT_VERSION=v1
REASONING="False"
model_base=/data_lyn/LLaVARadWeightNeoRed/vicuna-7b-v1.5
output_dir="/media/hdd2/MedEvalKit/Neo_Red/checkpoints"

# PROJECTOR="/PATH_TO/mm_projector.bin" # generated using pretrain.sh
vision_tower="biomedclip_cxr_518"
vision_tower_config="/data_lyn/LLaVARadWeightNeoRed/biomedclipcxr_518.json"
vision_tower_checkpoint="/data_lyn/LLaVARadWeightNeoRed/biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


# echo $CUDA_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES=0

################## Data ##################
#five public data:"/media/hdd2/MedEvalKit/LLaVARad/public_data/IU_XRAY/train.json","/media/hdd2/MedEvalKit/LLaVARad/public_data/MIMIC_CXR/train_with_ap_image.json"
# data_path={"/media/hdd2/MedEvalKit/LLaVARad/public_data/PATH_VQA/train-00003-of-00007-74b9b7b81cc55f90.parquet","/media/hdd2/MedEvalKit/LLaVARad/public_data/SLAKE/train.json","/media/hdd2/MedEvalKit/LLaVARad/public_data/VQA_RAD/train-00000-of-00001-eb8844602202be60.parquet"} # ,"/media/hdd2/MedEvalKit/LLaVARad/public_data/PATH_VQA/train-00003-of-00007-74b9b7b81cc55f90.parquet",
# hospital_data:
eval_data_path={"/data_lyn/data/NeoCXR/val.json"}
data_path={"/data_lyn/data/NeoCXR/train.json"}
loader="NeoCXRClinicToken" # Five  NeoCXRClinic(加cls loss clinicprompt) NeoCXRClinicToken(加cls loss clinicTokenprompt)
image_folder="/data_lyn/data/NeoCXR/images" # /media/hdd2/MedEvalKit/utils
################## Data ##################

################## Run name ##################
epoch="${2:-3}"
bsz="${3:-2}"
lr="1e-4"
experiment="NeoCXRClinic_3loss" #  NeoCXR_baseline_traindatanum_4395
schedule="lora-${epoch}e"
export run_name="${experiment}-${schedule}-${lr}"
# -$(date +%Y%m%d%H%M%S)
echo $run_name > run_name
################## Run name ##################

### loading projector
# PROJECTOR="/data_lyn/LLaVARadWeightNeoRed/non_lora_trainables.bin"


# --pretrain_mm_mlp_adapter ${PROJECTOR} \--master_port 29600
# Batch size is set for 4-GPU machines.
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-ft-${run_name}" WANDB_RUN_GROUP=fine-tune \
    deepspeed /data_lyn/Neo_Red/llava/train/train_mem.py \
    --deepspeed /data_lyn/Neo_Red/scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 32 \
    --lora_r 8 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --eval_data_path ${eval_data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --max_image_num 1 \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 2 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --num_diseases 9 \
    --cls_loss_weight 0.5 \
    --clinic_loss_weight 0.2 \
    --consist_loss_weight 0.2 \
    --report_to wandb
