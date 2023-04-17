# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.optimizer as optim

import models
import utils
from dataset import ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from loss import IBOTLoss
from models import IBOTHead, MultiCropWrapper
from transforms import IBOTAugmentation


def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
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
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for ibot
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
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
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
    parser.add_argument('--resume_path', default="", help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=10, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    return parser

def train_ibot(args):
    # ============ distributed env prepare ============
    dist.init_parallel_env()
    utils.fix_random_seeds(args.seed)
    
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # ============ preparing data  ============
    transform = IBOTAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset = ImageFolderMask(
        args.data_path, 
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch)
    sampler = paddle.io.DistributedBatchSampler(
        dataset, args.batch_size_per_gpu, shuffle=True, drop_last=True
    )
    data_loader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks  ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student, 
        IBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        )
    )
    teacher = MultiCropWrapper(
        teacher,
        IBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        )
    )
    # vit_s8 and vit_s16 are batch norm free models. here, we don't check bn
    # if utils.has_batchnorms(student):
        # student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        # teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    teacher = paddle.DataParallel(teacher)
    teacher_without_ddp = teacher._layers
    
    student = paddle.DataParallel(student)
    # teacher and student start with the same weights
    teacher_without_ddp.load_dict(student.state_dict())
    
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.stop_gradient = True
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = IBOTLoss(
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
    )

    # ============ preparing optimizer ============
    params_groups = utils.get_params_groups(student)
    clip = paddle.nn.ClipGradByGlobalNorm(args.clip_grad) if args.clip_grad != 0 else None
    opt = paddle.optimizer.AdamW(learning_rate=args.lr, parameters=params_groups, grad_clip=clip)
    # opt = paddle.optimizer.SGD(parameters=params_groups)
    fp16_scaler = None
    if args.use_fp16:
        # be consistent with pytorch default value.
        fp16_scaler = paddle.amp.GradScaler(init_loss_scaling=65536.0, incr_every_n_steps=2000)

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size_per_gpu * dist.get_world_size() / 256,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ============
    to_restore = {"epoch": 0}
    if args.resume_path != "":
        utils.restart_from_checkpoint(
            args.resume_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=opt,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]
    start_time = time.time()

    # ============ training ============
    print("Starting IBOT training!")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, opt, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args
        )

        # ============ check point save ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        # save on master
        if dist.get_rank() == 0:
            path = os.path.join(args.output_dir, f'ibot_{args.arch}_pretrain_checkpoint.pdparams')
            paddle.save(save_dict, path)
        
        if epoch == args.epochs or epoch % args.saveckp_freq == 0:
            if dist.get_rank() == 0:
                path = os.path.join(args.output_dir, f'ibot_{args.arch}_pretrain_checkpoint{epoch:04}.pdparams')
                paddle.save(save_dict, path)

        epoch_end_time = time.time()
        used_time = f"{(epoch_end_time - epoch_start_time) / 3600:.6f} h"

        # ============ write train log ============
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'used_time': used_time}
        print(log_stats)
        # is a master?
        if dist.get_rank() == 0:
            log_path = os.path.join(args.output_dir, "pretrained_log.txt")
            with open(log_path, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(
        student, teacher, teacher_without_ddp, ibot_loss, data_loader,
        optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
        fp16_scaler, args):

    metric_logger = utils.MetricLogger(" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.sublayers()[0].named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
    
    pred_labels, real_labels = [], []
    for it, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate
        # compute global training iteration
        it = len(data_loader) * epoch + it # global training iteration
        for i, param_group in enumerate(optimizer._param_groups):
            optimizer.set_lr(lr_schedule[it])
            # print(it,lr_schedule[it])
            if i == 0: # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        with paddle.amp.auto_cast(fp16_scaler is not None):
            # forward and compute ibot loss
            teacher_output = teacher(images[:args.global_crops_number])  # only the 2 global views pass through the teacher
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number]) # all views pass through the student
            
            student.sublayers()[0].backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
            student.sublayers()[0].backbone.masked_im_modeling = args.use_masked_im_modeling

            # print(student_output)
            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop('loss')

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)

        pred1 = utils.concat_all_gather(paddle.argmax(probs1[0],axis=1))
        pred2 = utils.concat_all_gather(paddle.argmax(probs2[1],axis=1))

        # 对齐metrics
        # print(pred1 == pred2)
        # print(paddle.sum(pred1 == pred2),pred1.shape[0])
        # print(pred1.shape,pred2.shape,type(pred1),pred1)
        # import numpy as np
        # fake_pred1 = np.random.randint(low=1,high=20,size=64).astype(np.int64)
        # fake_pred2 = np.random.randint(low=1,high=20,size=64).astype(np.int64)
        # fake_pred1 = paddle.to_tensor(fake_pred1)
        # fake_pred2 = paddle.to_tensor(fake_pred2)
        # print(pred1,pred2)
        acc = ((pred1 == pred2).sum())/ pred1.shape[0]
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.cuda()))

        # student update
        optimizer.clear_grad()
        if fp16_scaler is None:
            loss.backward()
            # if args.clip_grad:
            #     param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad != 0:
                fp16_scaler.unscale_(optimizer)
                # param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with paddle.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                new_val = m * param_k.numpy() + (1 - m) * param_q.detach().numpy()
                param_k.set_value(new_val)


        paddle.device.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer._learning_rate)
        metric_logger.update(wd=optimizer._param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    pred_labels = paddle.concat(pred_labels).cpu().detach().numpy()
    real_labels = paddle.concat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    
    # gather the stats from all processes
    if dist.is_initialized():
        metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    
    return return_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)