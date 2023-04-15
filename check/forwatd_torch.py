# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from PIL import Image
import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper
import paddle
import sys
import argparse
import time
import math
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import utils
from check.torch_utils import Torch_MultiCropWrapper, iBOTLoss, get_params_groups
import check.torch_vision_transformer as models
from check.torch_head import iBOTHead

def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_large'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
                        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
                        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
                        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default="/data3/linkaihao/reproduce/IBOT/pretrained", help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/data3/linkaihao/reproduce/IBOT-Paddle/check/duiqi", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser




def for_pass(args):
    setup_seed(10)
    student = models.__dict__["vit_small"](
        patch_size=16,
        drop_path_rate=0.1,
        return_all_tokens=True,
        masked_im_modeling=True,
    )
    teacher = models.__dict__["vit_small"](
        patch_size=16,
        return_all_tokens=True,
    )
    embed_dim = student.embed_dim

    student = Torch_MultiCropWrapper(student, iBOTHead(
        embed_dim,
        out_dim=8192,
        patch_out_dim=8192,
        norm=None,
        act="gelu",
        norm_last_layer=False,
        shared_head=True,
    ))

    teacher = Torch_MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            out_dim=8192,
            patch_out_dim=8192,
            norm=None,
            act="gelu",
            shared_head=True,
        ),
    )

    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    fp16_scaler = torch.cuda.amp.GradScaler()

    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    # open checkpoint file
    checkpoint = torch.load(os.path.join(args.load_from,"checkpoint.pth"), map_location="cpu")
    checkpoint["student"] = {k.replace("module.", ""): v for k, v in checkpoint["student"].items()}
    d = checkpoint["student"]
    print(d.keys())
    d.pop('head.last_layer.weight_g')
    print(d.keys())
    d.pop('head.last_layer2.weight_g')

    d = checkpoint["teacher"]
    print(d.keys())
    d.pop('head.last_layer.weight_g')
    print(d.keys())
    d.pop('head.last_layer2.weight_g')
    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    # kwargs={student=student,
    # #     teacher=teacher,
    # #     optimizer=optimizer,
    # #     fp16_scaler=fp16_scaler,
    # #     ibot_loss=ibot_loss,}
    # print(checkpoint["student"].keys())

    # transfer(checkpoint["teacher"], out_duiqi)
    msg = student.load_state_dict(checkpoint["student"], strict=False)
    # torch.save(student.state_dict(),os.path.join(args.load_from,"student_pretrain.pth"))
    print(msg)
    msg = teacher.load_state_dict(checkpoint["teacher"], strict=False)
    # torch.save(teacher.state_dict(),os.path.join(args.load_from,"teacher_pretrain.pth"))
    print(msg)
    student = student.cuda()
    teacher = teacher.cuda()
    student.eval()
    teacher.eval()

    torch_fake_data_list = []
    torch_fake_label_list = []
    torch_fake_mask_list = []

    paddle_fake_data_list = []
    paddle_fake_label_list = []
    paddle_fake_mask_list = []

    # np.random
    fake_data = np.random.rand(32, 3, 224, 224).astype(np.float32) - 0.5
    fake_mask = np.random.rand(32, 14, 14).astype(np.float32)
    fake_label = np.arange(1).astype(np.int64)

    torch_fake_data = torch.from_numpy(fake_data)
    torch_fake_mask = torch.from_numpy(fake_mask) > 0.5
    # paddle_fake_data = paddle.to_tensor(fake_data)
    # paddle_fake_mask = paddle.to_tensor(fake_mask) > 0.5

    fake_label = np.arange(1).astype(np.int64)
    for _ in range(0, 12):
        torch_fake_data_list.append(torch_fake_data.cuda())
        # fake_label_list.append(fake_label)
        torch_fake_mask_list.append(torch_fake_mask.cuda())
        # paddle_fake_data_list.append(paddle_fake_data)
        # fake_label_list.append(fake_label)
        # paddle_fake_mask_list.append(paddle_fake_mask)



    with torch.no_grad():
        reprod_logger = ReprodLogger()
        out_save = student(torch_fake_data_list, torch_fake_mask_list)
        reprod_logger.add("logits", out_save[0].cpu().detach().numpy())
        reprod_logger.save(os.path.join(args.output_dir,"forward_torch.npy"))
        out_save = teacher(torch_fake_data_list)
        reprod_logger = ReprodLogger()
        reprod_logger.add("logits", out_save[0].cpu().detach().numpy())
        reprod_logger.save(os.path.join(args.output_dir,"teacher_forward_torch.npy"))
        teacher_output = teacher(torch_fake_data_list[:2])
        student_output = student(torch_fake_data_list[:2],mask=torch_fake_mask_list[:2])
        student.backbone.masked_im_modeling = False
        student_local_cls = student(torch_fake_data_list[2:])[0] if len(torch_fake_data_list)>2 else None

        all_loss = ibot_loss(student_output, teacher_output, student_local_cls, torch_fake_mask_list[2:], 2)
        loss = all_loss.pop('loss')
        # print(loss)
        reprod_logger = ReprodLogger()
        reprod_logger.add("logits", loss.cpu().detach().numpy())
        reprod_logger.save(os.path.join(args.output_dir,"loss_torch.npy"))
    # print(msg)
    # msg = teacher.load_state_dict(checkpoint["teacher"], strict=False)
    # print(msg)
    # msg = optimizer.load_state_dict(checkpoint["optimizer"])
    # print(msg)
    # msg = fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
    # print(msg)
    # msg = ibot_loss.load_state_dict(checkpoint["ibot_loss"], strict=False)
    # print(msg)


def setup_seed(seed=10):
    import torch
    import os
    import numpy as np
    import random
    import paddle
    torch.manual_seed(seed)  # 为CPU设置随机种子
    paddle.seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        paddle.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

# def gen_fake_data():
#     fake_data_list = []
#     fake_label_list = []
#     fake_data = np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
#     fake_label = np.arange(1).astype(np.int64)
#     np.save("fake_data.npy", fake_data)
#     np.save("fake_label.npy", fake_label)


# def transfer(torch_dict,output_fp):
#     # torch_dict = torch.load(input_fp)
#     paddle_dict = {}
#     fc_names = [
#         'backbone.cls_token', 'backbone.pos_embed', 'backbone.masked_embed', 'backbone.patch_embed.proj.weight', 'backbone.patch_embed.proj.bias', 'backbone.blocks.0.norm1.weight',
#         'backbone.blocks.0.norm1.bias', 'backbone.blocks.0.attn.qkv.weight', 'backbone.blocks.0.attn.qkv.bias', 'backbone.blocks.0.attn.proj.weight',
#         'backbone.blocks.0.attn.proj.bias', 'backbone.blocks.0.norm2.weight', 'backbone.blocks.0.norm2.bias',
#         'backbone.blocks.0.mlp.fc1.weight', 'backbone.blocks.0.mlp.fc1.bias', 'backbone.blocks.0.mlp.fc2.weight',
#         'backbone.blocks.0.mlp.fc2.bias', 'backbone.blocks.1.norm1.weight', 'backbone.blocks.1.norm1.bias', 'backbone.blocks.1.attn.qkv.weight',
#         'backbone.blocks.1.attn.qkv.bias', 'backbone.blocks.1.attn.proj.weight', 'backbone.blocks.1.attn.proj.bias', 'backbone.blocks.1.norm2.weight',
#         'backbone.blocks.1.norm2.bias', 'backbone.blocks.1.mlp.fc1.weight', 'backbone.blocks.1.mlp.fc1.bias', 'backbone.blocks.1.mlp.fc2.weight', 'backbone.blocks.1.mlp.fc2.bias',
#         'backbone.blocks.2.norm1.weight', 'backbone.blocks.2.norm1.bias', 'backbone.blocks.2.attn.qkv.weight', 'backbone.blocks.2.attn.qkv.bias', 'backbone.blocks.2.attn.proj.weight',
#         'backbone.blocks.2.attn.proj.bias', 'backbone.blocks.2.norm2.weight', 'backbone.blocks.2.norm2.bias', 'backbone.blocks.2.mlp.fc1.weight', 'backbone.blocks.2.mlp.fc1.bias', 'backbone.blocks.2.mlp.fc2.weight',
#         'backbone.blocks.2.mlp.fc2.bias', 'backbone.blocks.3.norm1.weight', 'backbone.blocks.3.norm1.bias', 'backbone.blocks.3.attn.qkv.weight', 'backbone.blocks.3.attn.qkv.bias', 'backbone.blocks.3.attn.proj.weight',
#         'backbone.blocks.3.attn.proj.bias', 'backbone.blocks.3.norm2.weight', 'backbone.blocks.3.norm2.bias', 'backbone.blocks.3.mlp.fc1.weight', 'backbone.blocks.3.mlp.fc1.bias', 'backbone.blocks.3.mlp.fc2.weight',
#         'backbone.blocks.3.mlp.fc2.bias', 'backbone.blocks.4.norm1.weight', 'backbone.blocks.4.norm1.bias', 'backbone.blocks.4.attn.qkv.weight', 'backbone.blocks.4.attn.qkv.bias', 'backbone.blocks.4.attn.proj.weight',
#         'backbone.blocks.4.attn.proj.bias', 'backbone.blocks.4.norm2.weight', 'backbone.blocks.4.norm2.bias', 'backbone.blocks.4.mlp.fc1.weight', 'backbone.blocks.4.mlp.fc1.bias', 'backbone.blocks.4.mlp.fc2.weight',
#         'backbone.blocks.4.mlp.fc2.bias', 'backbone.blocks.5.norm1.weight', 'backbone.blocks.5.norm1.bias', 'backbone.blocks.5.attn.qkv.weight', 'backbone.blocks.5.attn.qkv.bias', 'backbone.blocks.5.attn.proj.weight',
#         'backbone.blocks.5.attn.proj.bias', 'backbone.blocks.5.norm2.weight', 'backbone.blocks.5.norm2.bias', 'backbone.blocks.5.mlp.fc1.weight', 'backbone.blocks.5.mlp.fc1.bias', 'backbone.blocks.5.mlp.fc2.weight',
#         'backbone.blocks.5.mlp.fc2.bias', 'backbone.blocks.6.norm1.weight', 'backbone.blocks.6.norm1.bias', 'backbone.blocks.6.attn.qkv.weight', 'backbone.blocks.6.attn.qkv.bias', 'backbone.blocks.6.attn.proj.weight',
#         'backbone.blocks.6.attn.proj.bias', 'backbone.blocks.6.norm2.weight', 'backbone.blocks.6.norm2.bias', 'backbone.blocks.6.mlp.fc1.weight', 'backbone.blocks.6.mlp.fc1.bias', 'backbone.blocks.6.mlp.fc2.weight',
#         'backbone.blocks.6.mlp.fc2.bias', 'backbone.blocks.7.norm1.weight', 'backbone.blocks.7.norm1.bias', 'backbone.blocks.7.attn.qkv.weight', 'backbone.blocks.7.attn.qkv.bias', 'backbone.blocks.7.attn.proj.weight',
#         'backbone.blocks.7.attn.proj.bias', 'backbone.blocks.7.norm2.weight', 'backbone.blocks.7.norm2.bias', 'backbone.blocks.7.mlp.fc1.weight', 'backbone.blocks.7.mlp.fc1.bias', 'backbone.blocks.7.mlp.fc2.weight',
#         'backbone.blocks.7.mlp.fc2.bias', 'backbone.blocks.8.norm1.weight', 'backbone.blocks.8.norm1.bias', 'backbone.blocks.8.attn.qkv.weight', 'backbone.blocks.8.attn.qkv.bias', 'backbone.blocks.8.attn.proj.weight',
#         'backbone.blocks.8.attn.proj.bias', 'backbone.blocks.8.norm2.weight', 'backbone.blocks.8.norm2.bias', 'backbone.blocks.8.mlp.fc1.weight', 'backbone.blocks.8.mlp.fc1.bias', 'backbone.blocks.8.mlp.fc2.weight',
#         'backbone.blocks.8.mlp.fc2.bias', 'backbone.blocks.9.norm1.weight', 'backbone.blocks.9.norm1.bias', 'backbone.blocks.9.attn.qkv.weight', 'backbone.blocks.9.attn.qkv.bias', 'backbone.blocks.9.attn.proj.weight',
#         'backbone.blocks.9.attn.proj.bias', 'backbone.blocks.9.norm2.weight', 'backbone.blocks.9.norm2.bias', 'backbone.blocks.9.mlp.fc1.weight', 'backbone.blocks.9.mlp.fc1.bias', 'backbone.blocks.9.mlp.fc2.weight',
#         'backbone.blocks.9.mlp.fc2.bias', 'backbone.blocks.10.norm1.weight', 'backbone.blocks.10.norm1.bias', 'backbone.blocks.10.attn.qkv.weight', 'backbone.blocks.10.attn.qkv.bias', 'backbone.blocks.10.attn.proj.weight',
#         'backbone.blocks.10.attn.proj.bias', 'backbone.blocks.10.norm2.weight', 'backbone.blocks.10.norm2.bias', 'backbone.blocks.10.mlp.fc1.weight', 'backbone.blocks.10.mlp.fc1.bias', 'backbone.blocks.10.mlp.fc2.weight',
#         'backbone.blocks.10.mlp.fc2.bias', 'backbone.blocks.11.norm1.weight', 'backbone.blocks.11.norm1.bias', 'backbone.blocks.11.attn.qkv.weight', 'backbone.blocks.11.attn.qkv.bias', 'backbone.blocks.11.attn.proj.weight',
#         'backbone.blocks.11.attn.proj.bias', 'backbone.blocks.11.norm2.weight', 'backbone.blocks.11.norm2.bias', 'backbone.blocks.11.mlp.fc1.weight', 'backbone.blocks.11.mlp.fc1.bias', 'backbone.blocks.11.mlp.fc2.weight',
#         'backbone.blocks.11.mlp.fc2.bias', 'backbone.norm.weight', 'backbone.norm.bias', 'head.mlp.0.weight', 'head.mlp.0.bias', 'head.mlp.2.weight', 'head.mlp.2.bias', 'head.mlp.4.weight', 'head.mlp.4.bias', 'head.last_layer.weight_g',
#         'head.last_layer.weight_v', 'head.last_layer2.weight_g', 'head.last_layer2.weight_v']
#
#     for key in torch_dict:
#         weight = torch_dict[key].cpu().detach().numpy()
#         flag = [i in key for i in fc_names]
#         if any(flag):
#             print("weight {} need to be trans".format(key))
#             weight = weight.transpose()
#         paddle_dict[key] = weight
#     paddle.save(paddle_dict, output_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for_pass(args)
    # train_ibot(args)