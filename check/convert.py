import torch
import paddle

from collections import OrderedDict

from models import vit_small, MultiCropWrapper, IBOTHead


student = vit_small(patch_size=16, drop_path_rate=0.1, return_all_tokens=True, masked_im_modeling=False)
teacher = vit_small(patch_size=16, return_all_tokens=True)
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

torch_ckpt = torch.load("/home/xiejunlin/workspace/ibot/pretrained/checkpoint.pth", map_location='cpu')
torch_weight = torch_ckpt['teacher']
torch_weight = {k.replace("module.", ""): v for k, v in torch_weight.items()}
torch_weight.pop("head.last_layer.weight_g")
torch_weight.pop("head.last_layer2.weight_g")

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

# paddle.save(new_weight_dict, "/home/xiejunlin/workspace/ibot/pretrained/checkpoint_teacher.pdparams")

stu = paddle.load("check/ckpt/student.pdparams")

optimizer = torch_ckpt['optimizer']
# new_optim = OrderedDict()
# for k, v in optimizer.items():
#     if k == "state":
#         new_optim['state'] = v
#     else:
        
ibot_loss = torch_ckpt['ibot_loss']
ibot_loss = OrderedDict({
    k: v.detach().numpy() for k, v in ibot_loss.items()
})

paddle.save({"student": stu, "teacher":new_weight_dict, "epoch": 100, "ibot_loss": ibot_loss}, "check/ckpt/full_ckpt.pdparams")

