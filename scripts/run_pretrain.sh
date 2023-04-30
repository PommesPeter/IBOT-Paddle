#!/usr/bin/env bash

MASTER_ADDR=121.48.161.104:52100
NUM_NODES=2

python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" --master=127.0.0.1:52100 --nnodes=$NUM_NODES --nproc_per_node=8 main_ibot.py \
    --act_in_head 'gelu' \
    --arch 'vit_small' \
    --batch_size_per_gpu 64 \
    --clip_grad 3.0 \
    --data_path '~/data/datasets/imagenet/mini/train' \
    --epochs 800 \
    --freeze_last_layer 1 \
    --global_crops_number 2 \
    --global_crops_scale 0.25 1.0 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --local_crops_number 10 \
    --local_crops_scale 0.05 0.25 \
    --local_rank 0 \
    --lr 0.0005 \
    --min_lr 1e-06 \
    --momentum_teacher 0.996 \
    --norm_last_layer False \
    --num_workers 10 \
    --optimizer 'adamw' \
    --out_dim 8192 \
    --output_dir './work_dirs/ibot_vits_p16_pretrain' \
    --patch_out_dim 8192 \
    --patch_size 16 \
    --pred_ratio 0.0 0.3 \
    --pred_ratio_var 0.0 0.2 \
    --pred_shape 'block' \
    --pred_start_epoch 0 \
    --saveckp_freq 40 \
    --seed 0 \
    --shared_head True \
    --teacher_patch_temp 0.07 \
    --teacher_temp 0.07 \
    --use_fp16 True \
    --use_masked_im_modeling True \
    --warmup_epochs 10 \
    --warmup_teacher_patch_temp 0.04 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --window_size 7
