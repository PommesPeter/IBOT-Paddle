import numpy as np
import paddle
import sys
sys.path.append("../")
sys.path.append("./")
import torch
from reprod_log import ReprodLogger
from collections import OrderedDict
from loss import IBOTLoss
from models.ibot import IBOT_ViT_small_patch16_224
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


if __name__ == "__main__":
    paddle.set_device("gpu")
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

    student = IBOT_ViT_small_patch16_224(pretrained=False,norm_last_layer=False,masked_im_modeling=True)
    teacher = IBOT_ViT_small_patch16_224(pretrained=False,norm_last_layer=True,masked_im_modeling=False)
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

