python -m paddle.distributed.launch --gpus="0,7" evaluation/linear/eval_linear.py \
    --patch_size 16 
    --data_path /home/xiejunlin/data/datasets/imagenet/mini
    --pretrained_weights /home/xiejunlin/workspace/IBOT-Paddle/output/ibot_vit_small_pretrain_checkpoint0400.pdparams 
    --checkpoint_key teacher 
    --batch_size_per_gpu 128 
    --lr 0.01 
    --output_dir ./output_linear_mini


