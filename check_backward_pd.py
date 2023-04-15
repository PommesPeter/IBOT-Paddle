import pickle

import numpy as np
import paddle
from utils import get_args_from, get_args_parser
from reprod_log import ReprodLogger
from models.ibot import vit_small

import utils
from loss import IBOTLoss
from models.ibot import IBOTHead, MultiCropWrapper
from models.linear import LinearClassifier

paddle.seed(10)

# with open("../data/8_images.pkl", "rb") as f:
#     images: list = pickle.load(f)
# images = [paddle.to_tensor(it) for it in images]

# convert to tensor
batch_data = np.load("/home/xiejunlin/workspace/ibot/batch_data.npy", allow_pickle=True)
images = [paddle.to_tensor(it) for it in batch_data[0]]
label = paddle.to_tensor(batch_data[1])
masks = [paddle.to_tensor(it) for it in batch_data[2]]
batch_data = tuple((images, label, masks))
LEN_DATALOADER = 1


args = get_args_from("check/args.json", get_args_parser())
args.epochs = 40

# only support vit_s16 currently
student_bkb = vit_small(
    patch_size=args.patch_size,
    drop_path_rate=args.drop_path,
    return_all_tokens=True,
    masked_im_modeling=args.use_masked_im_modeling,
)
teacher_bkb = vit_small(
    patch_size=args.patch_size,
    return_all_tokens=True,
)
embed_dim = student_bkb.embed_dim

print(f"Student and Teacher are built: they are both {args.arch} network.")

student = MultiCropWrapper(
    student_bkb, IBOTHead(
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
    teacher_bkb,
    IBOTHead(
        embed_dim, 
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        shared_head=args.shared_head_teacher,
    )
)
teacher_without_ddp = teacher
# teacher and student start with the same weights
teacher_without_ddp.load_dict(student.state_dict())

for p in teacher.parameters():
    p.requires_grad = False

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
optimizer = paddle.optimizer.AdamW(learning_rate=args.lr, parameters=params_groups)

# ============ init schedulers ... ============
lr_schedule = utils.cosine_scheduler(
    args.lr * args.batch_size_per_gpu / 256,  # linear scaling rule
    args.min_lr,
    args.epochs, LEN_DATALOADER,
    warmup_epochs=args.warmup_epochs,
)
wd_schedule = utils.cosine_scheduler(
    args.weight_decay,
    args.weight_decay_end,
    args.epochs, LEN_DATALOADER,
)
# momentum parameter is increased to 1. during training with a cosine schedule
momentum_schedule = utils.cosine_scheduler(
    args.momentum_teacher, 1,
    args.epochs, LEN_DATALOADER
)
print("Loss, optimizer and schedulers ready.")

# ============ optionally resume training ============
to_restore = {"epoch": 0}
utils.restart_from_checkpoint(
    "/home/xiejunlin/workspace/IBOT-Paddle/check/ckpt/full_ckpt.pdparams",
    run_variables=to_restore,
    student=student,
    teacher=teacher,
    optimizer=optimizer, # todo: need to load optimizer parameters
    ibot_loss=ibot_loss
)
start_epoch = 0


def log_state():
    bkb_log = ReprodLogger()
    i = 0
    for k, v in teacher_bkb.state_dict().items():
        if k.find("attn.proj.weight") != -1:
            bkb_log.add(f"tea_state_{i}", v.numpy().T)
        else:
            bkb_log.add(f"tea_state_{i}", v.numpy())
        i += 1

    i = 0
    for k, v in student_bkb.state_dict().items():
        if k.find("attn.proj.weight") != -1:
            bkb_log.add(f"stu_state_{i}", v.numpy().T)
        else:
            bkb_log.add(f"stu_state_{i}", v.numpy())
        i += 1
    bkb_log.save("./reprod_out/pd_bkb_state.npy")


def log_lr():
    lr_log = ReprodLogger()
    lr_log.add("lr", lr_schedule)
    lr_log.add("wd", wd_schedule)
    lr_log.add("momentum_schedule", momentum_schedule)
    lr_log.save("./reprod_out/lr_pd.npy")


def backbone_one_epoch(epoch, data, reprod_log):
    images, labels, masks = data
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
    
    cur_iter_num = LEN_DATALOADER * epoch + 0

    optimizer.set_lr(lr_schedule[cur_iter_num])
    for i, param_group in enumerate(optimizer._param_groups):
        # if i == 0:
            # only the first group is regularized
            # param_group["weight_decay"] = wd_schedule[cur_iter_num]
        pass
    
    teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
    student_output = student(images[:2], mask=masks[:2])      # all views pass through the student

    student.backbone.masked_im_modeling = False
    student_local_cls = student(images[2:])[0] if len(images) > 2 else None
    student.backbone.masked_im_modeling = args.use_masked_im_modeling
    
    # for i in range(2):
    #     arr = teacher_bkb(data[i]).detach().numpy()
    #     reprod_log.add(f"bkb_fea_{i}", arr)

    # for i in range(2, 12):
    #     arr = student_bkb(data[i]).detach().numpy()
    #     reprod_log.add(f"bkb_stu_{i}", arr)

    for i in range(len(teacher_output)):
        arr = teacher_output[i].detach().numpy()
        reprod_log.add(f"tea_out_{i}", arr)

    for i in range(len(student_output)):
        arr = student_output[i].detach().numpy()
        reprod_log.add(f"stu_out_{i}", arr)

    all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
    loss = all_loss.pop("loss")
    cls_loss = all_loss.pop("cls")
    patch_loss = all_loss.pop("patch")
    
    reprod_log.add(f"center_epoch_{epoch}", ibot_loss.center.numpy())
    reprod_log.add(f"loss_epoch_{epoch}", loss.detach().numpy())
    reprod_log.add(f"cls_loss_epoch_{epoch}", cls_loss.detach().numpy())
    reprod_log.add(f"patch_loss_epoch_{epoch}", patch_loss.detach().numpy())

    # student update
    loss.backward()
    utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)

    # log grad
    # iter = student.parameters().__iter__()
    # idx = 0
    # for p in iter:
    #     if not p.stop_gradient:
    #         print("idx", idx)
    #         reprod_log.add(f"grad_{idx}_epoch_{epoch}", p.grad.numpy())
    #         idx += 1

    optimizer.step()
    optimizer.clear_grad()

    # EMA update for the teacher
    with paddle.no_grad():
        m = momentum_schedule[cur_iter_num]
        for param_q, param_k in zip(params_q, params_k):
            new_val = m * param_k.numpy() + (1 - m) * param_q.numpy()
            param_k.set_value(new_val)
            
    return loss.item()


def check_backbone_backward():
    student.eval()
    teacher.eval()
    teacher_without_ddp.eval()

    reprod_log = ReprodLogger()
    for epoch in range(10):
        loss = backbone_one_epoch(epoch, batch_data, reprod_log)
        print(f"epoch: {epoch}, loss: {loss}")

    reprod_log.save("./reprod_out/pd_backward.npy")


# ============= Linear Probing =============
# num_labels = 1000
# lr = 0.01
# batch_size = 1024
# epochs = 40
# last_blocks = 4

# linear_clf = LinearClassifier(embed_dim * last_blocks, num_labels=num_labels)
# # label = np.load("../data/8_labels.npy")
# # label = paddle.to_tensor(label)
# lr = lr * (batch_size / 256)
# scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epochs, eta_min=0)
# lin_opt = paddle.optimizer.Momentum(
#     parameters=linear_clf.parameters(),
#     learning_rate=scheduler,
#     momentum=0.9,
#     weight_decay=0.0
# )
# model = vit_small()
# utils.load_pretrained_weights(model, "teacher", "ckpt/ibot_vitsmall16_full_pretrain.pdparams")

# state_dict = paddle.load("ckpt/ibot_vitsmall16_linearweights.pdparams")['state_dict']
# new_state_dict = {k.replace("module.", "") : v for k, v in state_dict.items()}
# linear_clf.load_dict(new_state_dict)
# model.eval()

# hook_module_idx = 0


# def check_sub_m_out():
#     """检测backbone的每个子模块的输出"""
#     main_log = ReprodLogger()

#     def one_clf_epoch(epoch, data):
#         image = data[0]

#         # forward
#         with paddle.no_grad():
#             intermediate_output = model.get_intermediate_layers(image, last_blocks)
#             output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
#             main_log.add(f"output_epoch_{epoch}", output.numpy())

#     for epoch in range(5):
#         def hook(module, input, output):
#             global hook_module_idx
#             if isinstance(output, tuple):
#                 for it in output:
#                     main_log.add(f"module_{hook_module_idx}", it.detach().cpu().numpy())
#             else:
#                 main_log.add(f"module_{hook_module_idx}", output.detach().cpu().numpy())
#             hook_module_idx += 1

#         handles = []
#         # set hooks
#         handles.append(model.patch_embed.register_forward_post_hook(hook))
#         handles.append(model.blocks[0].register_forward_post_hook(hook))
#         handles.append(model.blocks[5].register_forward_post_hook(hook))
#         handles.append(model.blocks[11].register_forward_post_hook(hook))

#         one_clf_epoch(epoch, images)

#         # remove handle
#         for hd in handles:
#             hd.remove()

#     main_log.save("./reprod_out/sub_m_pd.npy")


# def check_clf_backward():
#     """检测分类层反向过程
#     收集：学习率，线性层输出，预测结果，损失，线性层梯度
#     """
#     lr_log = ReprodLogger()
#     main_log = ReprodLogger()

#     def one_clf_epoch(epoch, data):
#         linear_clf.train()
#         image = data[0]
#         lr_log.add(f"lr_epoch_{epoch}", np.array([lin_opt.get_lr()]))

#         # forward
#         with paddle.no_grad():
#             intermediate_output = model.get_intermediate_layers(image, last_blocks)
#             output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)

#         output = linear_clf(output)
#         main_log.add(f"output_epoch_{epoch}", output.numpy())
#         main_log.add(f"pred_epoch_{epoch}", output.numpy().max(axis=1))

#         # compute cross entropy loss
#         loss = paddle.nn.CrossEntropyLoss()(output, label)
#         print(f"epoch: {epoch}, loss: {loss.item()}")
#         main_log.add(f"loss_epoch_{epoch}", loss.numpy())

#         lin_opt.clear_grad()
#         loss.backward()

#         # grad
#         iter = linear_clf.parameters().__iter__()
#         idx = 0
#         for p in iter:
#             main_log.add(f"grad_{idx}_epoch_{epoch}", p.grad.numpy())
#             idx += 1

#         lin_opt.step()

#     for epoch in range(5):
#         one_clf_epoch(epoch, images)

#     lr_log.save("./reprod_out/clf_lr_pd.npy")
#     main_log.save("./reprod_out/clf_log_pd.npy")


if __name__ == "__main__":
    log_lr()
    # check_clf_backward()
    check_backbone_backward()
    # check_sub_m_out()