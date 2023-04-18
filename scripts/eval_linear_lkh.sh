python -m paddle.distributed.launch --gpus="0,7" evaluation/linear/eval_linear.py --patch_size 16 --data_path /data3/linkaihao/dataset/mini-imagenet-1k --pretrained_weights /data3/linkaihao/reproduce/IBOT-Paddle/check/duiqi/teacher.pdparams --checkpoint_key teacher --batch_size_per_gpu 128 --lr 0.01 --output_dir /data3/linkaihao/reproduce/IBOT-Paddle/output_linear_raw_imagenet


