import numpy as np
import paddle
import sys
sys.path.append("../")
sys.path.append("./")
import torch
from models import vit_small, MultiCropWrapper, IBOTHead


from reprod_log import ReprodLogger
from collections import OrderedDict
from loss import IBOTLoss
import argparse

def setup_seed(seed=10):
    import torch
    import os
    import numpy as np
    import random
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
        #os.environ['PYTHONHASHSEED'] = str(seed)

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
    parser.add_argument('--shared_head', default=False, type=bool, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=bool, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=bool,
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
    parser.add_argument('--use_masked_im_modeling', default=True, type=bool,
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
    parser.add_argument('--use_fp16', type=bool, default=True, help="""Whether or not
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
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
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
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

if __name__ == "__main__":
    paddle.set_device("gpu")
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    # load model
    # the model is save into ~/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth
    # def logger
    reprod_logger = ReprodLogger()
    setup_seed(10)

    # weight = []
    # for torch_key in torch_weight.keys():
    #     weight.append([torch_key, torch_weight[torch_key].detach().numpy()])
    #     print(torch_key)
    # print(weight[0])

    # student = IBOT_ViT_small_patch16_224(pretrained=False,norm_last_layer=False,masked_im_modeling=True)
    # teacher = IBOT_ViT_small_patch16_224(pretrained=False,norm_last_layer=True,masked_im_modeling=False)

    student = vit_small(patch_size=args.patch_size,drop_path_rate=args.drop_path,return_all_tokens=True, masked_im_modeling=args.use_masked_im_modeling)
    teacher = vit_small(patch_size=args.patch_size,  return_all_tokens=True)
    embed_dim = student.embed_dim
    student = MultiCropWrapper(student, IBOTHead(
        embed_dim,
        out_dim=8192,
        patch_out_dim=8192,
        norm=None,
        act="gelu",
        norm_last_layer=False,
        shared_head=True,
    ))

    teacher = MultiCropWrapper(
        teacher,
        IBOTHead(
            embed_dim,
            out_dim=8192,
            patch_out_dim=8192,
            norm=None,
            act="gelu",
            shared_head=True,
        ),
    )

    paddle_weight = student.state_dict()
    student.eval()
    teacher.eval()
    # 检查是否paddle中的key在torch的dict中能找到
    # for paddle_key in paddle_weight:
    #     if paddle_key in torch_weight.keys():
    #         print("Oh Yeah")
    #     else:
    #         print("No!!!")

    # param_dict = paddle.load("/data1/linkaihao/reproduce/ibot/duiqi/student2.pdparams")
    # paddle_model.set_dict(param_dict)
    torch_weight = torch.load("/data1/linkaihao/reproduce/ibot/duiqi/student_pretrain.pth")
    new_weight_dict = OrderedDict()
    for paddle_key in paddle_weight.keys():
        # 首先要确保torch的权重里面有这个key，这样就可以避免DIY模型中一些小模块影响权重转换
        if paddle_key in torch_weight.keys():
            # pytorch权重和paddle模型的权重为2维时需要转置，其余情况不需要
            if len(torch_weight[paddle_key].detach().numpy().shape) == 2 and "masked_embed" not in paddle_key:
                # print(paddle_key)
                new_weight_dict[paddle_key] = torch_weight[paddle_key].detach().numpy().T
            else:
                new_weight_dict[paddle_key] = torch_weight[paddle_key].detach().numpy()
        else:
            pass

    student.set_dict(new_weight_dict)

    torch_weight = torch.load("/data1/linkaihao/reproduce/ibot/duiqi/teacher_pretrain.pth")
    teacher_weight_dict = OrderedDict()
    for paddle_key in paddle_weight.keys():
        # 首先要确保torch的权重里面有这个key，这样就可以避免DIY模型中一些小模块影响权重转换
        if paddle_key in torch_weight.keys():
            # pytorch权重和paddle模型的权重为2维时需要转置，其余情况不需要
            if len(torch_weight[paddle_key].detach().numpy().shape) == 2 and "masked_embed" not in paddle_key:
                # print(paddle_key)
                teacher_weight_dict[paddle_key] = torch_weight[paddle_key].detach().numpy().T
            else:
                teacher_weight_dict[paddle_key] = torch_weight[paddle_key].detach().numpy()
        else:
            pass

    teacher.set_dict(teacher_weight_dict)
    ibot_loss = IBOTLoss(
        out_dim=8192,
        patch_out_dim=8192,
        ngcrops=2,
        nlcrops=10,
        teacher_temp=0.07
    )

    # paddle.save(paddle_model.state_dict(),"/data1/linkaihao/reproduce/ibot/duiqi/student2.pdparams")
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
    # np.save("fake_data.npy", fake_data)
    # np.save("fake_mask.npy", fake_label)

    # fake_data = paddle.randn([32, 3, 224, 224])
    # fake_mask = paddle.randn([32, 14, 14]) > 0.5
    # torch_fake_data = torch.rand(32, 3, 224, 224)
    # torch_fake_mask = torch.randn(32, 14, 14) > 0.5

    torch_fake_data = torch.from_numpy(fake_data)
    torch_fake_mask = torch.from_numpy(fake_mask) > 0.5
    paddle_fake_data = paddle.to_tensor(fake_data)
    paddle_fake_mask = paddle.to_tensor(fake_mask) > 0.5



    # fake_mask = np.random.rand(1, 3, 14, 14) > 0.5
    fake_label = np.arange(1).astype(np.int64)
    for _ in range(0, 12):
        torch_fake_data_list.append(torch_fake_data)
        # fake_label_list.append(fake_label)
        torch_fake_mask_list.append(torch_fake_mask)
        paddle_fake_data_list.append(paddle_fake_data)
        # fake_label_list.append(fake_label)
        paddle_fake_mask_list.append(paddle_fake_mask.cuda())




    with paddle.no_grad():
        reprod_logger = ReprodLogger()
        out_save = student(paddle_fake_data_list,mask=paddle_fake_mask_list)
        reprod_logger.add("logits", out_save[0].cpu().detach().numpy())
        reprod_logger.save("/data1/linkaihao/reproduce/ibot/duiqi/forward_paddle.npy")
        reprod_logger = ReprodLogger()
        out_save = teacher(paddle_fake_data_list)
        reprod_logger.add("logits", out_save[0].cpu().detach().numpy())
        reprod_logger.save("/data1/linkaihao/reproduce/ibot/duiqi/t_forward_paddle.npy")
        teacher_output = teacher(paddle_fake_data_list[:2])
        student_output = student(paddle_fake_data_list[:2], mask=paddle_fake_mask_list[:2])
        student.backbone.masked_im_modeling = False
        student_local_cls = student(paddle_fake_data_list[2:])[0] if len(paddle_fake_data_list) > 2 else None
        all_loss = ibot_loss(student_output, teacher_output, student_local_cls, paddle_fake_mask_list[2:], 2)
        loss = all_loss.pop('loss')
        print(loss.cpu().detach().numpy())
        reprod_logger = ReprodLogger()
        reprod_logger.add("logits", loss.cpu().detach().numpy())
        reprod_logger.save("/data1/linkaihao/reproduce/ibot/duiqi/loss_paddle.npy")

    from reprod_log import ReprodDiffHelper

    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("/data1/linkaihao/reproduce/ibot/duiqi/forward_torch.npy")
    paddle_info = diff_helper.load_info("/data1/linkaihao/reproduce/ibot/duiqi/forward_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="/data1/linkaihao/reproduce/ibot/duiqi/forward_diff.log")

    # diff_helper = ReprodDiffHelper()
    # torch_info = diff_helper.load_info("/data1/linkaihao/reproduce/ibot/duiqi/t_forward_torch.npy")
    # paddle_info = diff_helper.load_info("/data1/linkaihao/reproduce/ibot/duiqi/t_forward_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="/data1/linkaihao/reproduce/ibot/duiqi/forward_diff.log")

    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("/data1/linkaihao/reproduce/ibot/duiqi/loss_torch.npy")
    paddle_info = diff_helper.load_info("/data1/linkaihao/reproduce/ibot/duiqi/loss_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="/data1/linkaihao/reproduce/ibot/duiqi/forward_diff.log")

