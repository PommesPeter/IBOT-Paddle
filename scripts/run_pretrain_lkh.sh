# python -m paddle.distributed.launch --gpus="0,1" main_ibot.py \
#     --patch_size 16 \
#     --epochs 800 \
#     --arch vit_small \
#     --data_path ~/data/datasets/imagenet/mini/train \
#     --batch_size_per_gpu 32 \
#     --output_dir ./work_dirs/pretraine_lkh_no_pred_ratio_var \
#     --teacher_temp 0.07 \
#     --warmup_teacher_temp_epochs 30 \
#     --norm_last_layer False \
#     --shared_head True \
#     --global_crops_scale 0.25 1 \
#     --local_crops_number 10 \
#     --local_crops_scale 0.05 0.25 \
#     --pred_ratio 0 0.3 \
#     --lr 0.001 \
#     --out_dim 8192 \
#     # --pred_ratio_var 0 0.2 \

python -m paddle.distributed.launch --gpus="0,1" main_ibot.py \
    --patch_size 16 \
    --epochs 800 \
    --arch vit_small \
    --data_path ~/data/datasets/imagenet/mini/train \
    --batch_size_per_gpu 32 \
    --output_dir ./work_dirs/lkh_test \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 30 \
    --norm_last_layer False \
    --shared_head True \
    --global_crops_scale 0.25 1 \
    --local_crops_number 10 \
    --local_crops_scale 0.05 0.25 \
    --pred_ratio 0 0.3 \
    --lr 0.001 \
    --out_dim 8192 \
    --num_workers 10 \
    --saveckp_freq 20
    