# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import json
import os
from pathlib import Path

import paddle
import paddle.distributed as dist
import paddle.io as io
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as T

import models
import utils
from dataset import ImageFolder
from models import LinearClassifier


def eval_linear(args):
    dist.init_parallel_env()

    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    # fix the seed for reproducibility
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    if args.arch == "dalle_encoder":
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(112),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        val_transform = T.Compose(
            [
                T.Resize(128, interpolation=3),
                T.CenterCrop(112),
                T.ToTensor(),
            ]
        )
    else:
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        val_transform = T.Compose(
            [
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset_train = ImageFolder(train_dir, transform=train_transform)
    dataset_val = ImageFolder(val_dir, transform=val_transform)

    sampler = io.DistributedBatchSampler(dataset_train)
    train_loader = io.DataLoader(
        dataset_train,
        batch_sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
    )
    val_loader = io.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )

    # ============ building network ... ============
    if "swin" in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](
            window_size=args.window_size, patch_size=args.patch_size, num_classes=0
        )
        embed_dim = model.num_features
    else:
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens == 1,
        )
        embed_dim = model.embed_dim

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    # utils.load_pretrained_weights(
    #     model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    # )

    if "swin" in args.arch:
        num_features = []
        for i, d in enumerate(model.depths):
            num_features += [int(model.embed_dim * 2**i)] * d
        feat_dim = sum(num_features[-args.n_last_blocks :])
    else:
        feat_dim = embed_dim * (
            args.n_last_blocks * int(args.avgpool_patchtokens != 1)
            + int(args.avgpool_patchtokens > 0)
        )
    linear_clf = LinearClassifier(feat_dim, num_labels=args.num_labels)
    linear_clf = paddle.DataParallel(linear_clf)

    # set optimizer
    base_lr = args.lr * (
        args.batch_size * dist.get_world_size() / 256
    )  # linear scaling rule

    parameters = linear_clf.parameters()
    scheduler = optim.lr.CosineAnnealingDecay(
        learning_rate=base_lr, T_max=args.epochs, eta_min=0
    )
    optimizer = optim.Momentum(
        scheduler,
        parameters=parameters,
        momentum=0.9,
        weight_decay=0.0,  # we do not apply weight decay
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.0}
    if args.resume_path != "":
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.resume_path),
            run_variables=to_restore,
            state_dict=linear_clf,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"resume from epoch {to_restore['epoch']}")
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.eval()
        train_stats = train(
            model,
            linear_clf,
            optimizer,
            train_loader,
            epoch,
            args.n_last_blocks,
            args.avgpool_patchtokens,
        )
        scheduler.step()

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_clf.eval()
            test_stats = validate_network(
                val_loader,
                model,
                linear_clf,
                args.n_last_blocks,
                args.avgpool_patchtokens,
            )
            print(
                f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )
            log_stats = {
                **{k: v for k, v in log_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }

            if dist.get_rank() == 0:
                # always only save best checkpoint till now
                with open(Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_clf.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": test_stats["acc1"],
                }
                paddle.save(
                    save_dict,
                    os.path.join(
                        args.output_dir,
                        "checkpoint_{}_linear_{}.pth".format(
                            args.checkpoint_key, epoch
                        ),
                    ),
                )

            best_acc = max(best_acc, test_stats["acc1"])
            print(f"Max accuracy so far: {best_acc:.2f}%")

    print(
        "Training of the supervised linear classifier on frozen features completed.\n"
        "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc)
    )


def train(model, linear_clf, optimizer, loader, epoch, n, avgpool):
    linear_clf.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    for inp, target in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with paddle.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avgpool == 0:
                # norm(x[:, 0])
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool == 1:
                # x[:, 1:].mean(1)
                output = [paddle.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool == 2:
                # norm(x[:, 0]) + x[:, 1:].mean(1)
                output = [x[:, 0] for x in intermediate_output] + [
                    paddle.mean(intermediate_output[-1][:, 1:], dim=1)
                ]
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)

            output = paddle.cat(output, dim=-1)

        output = linear_clf(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.clear_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        if dist.is_initialized():
            paddle.device.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer._learning_rate.last_lr)

    # gather the stats from all processes
    if dist.is_initialized():
        metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def validate_network(val_loader, model, linear_clf, n, avgpool):
    linear_clf.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with paddle.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avgpool == 0:
                # norm(x[:, 0])
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool == 1:
                # x[:, 1:].mean(1)
                output = [paddle.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool == 2:
                # norm(x[:, 0]) + x[:, 1:].mean(1)
                output = [x[:, 0] for x in intermediate_output] + [
                    paddle.mean(intermediate_output[-1][:, 1:], dim=1)
                ]
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)

            output = paddle.cat(output, dim=-1)

        output = linear_clf(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if args.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            (acc1,) = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        if args.num_labels >= 5:
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if args.num_labels >= 5:
        print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss,
            )
        )
    else:
        print(
            "* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1, losses=metric_logger.loss
            )
        )

    # gather the stats from all processes
    if dist.is_initialized():
        metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluation with linear classification on ImageNet"
    )
    parser.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=0,
        choices=[0, 1, 2],
        type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base/large-size models with [CLS] token when doing linear classification.""",
    )
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_large",
            "swin_tiny",
            "swin_small",
            "swin_base",
            "swin_large",
            "resnet50",
            "resnet101",
            "dalle_encoder",
        ],
        help="Architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--window_size",
        default=7,
        type=int,
        help="Window size of the model (only swin).",
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""",
    )
    parser.add_argument(
        "--pretrained_linear",
        default="",
        type=str,
        help="Path to pretrained linear clf weights.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--data_path",
        default="~/data/datasets/imagenet/mini",
        type=str,
        help="Please specify path to the ImageNet data.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--val_freq", default=1, type=int, help="Epoch frequency for validation."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path to save logs and checkpoints"
    )
    parser.add_argument(
        "--num_labels",
        default=1000,
        type=int,
        help="Number of labels for linear classifier",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path to load checkpoints to resume training",
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(","):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
