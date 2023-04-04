# Image BERT Pre-Training with iBOT <img width="32" alt="iBOT Icon" src=".github/ibot.png">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ibot-image-bert-pre-training-with-online/unsupervised-image-classification-on-imagenet)](https://paperswithcode.com/sota/unsupervised-image-classification-on-imagenet?p=ibot-image-bert-pre-training-with-online) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ibot-image-bert-pre-training-with-online/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=ibot-image-bert-pre-training-with-online) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ibot-image-bert-pre-training-with-online/self-supervised-image-classification-on-1)](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?p=ibot-image-bert-pre-training-with-online)

Paddle implementation code for paper "iBOT 🤖: Image BERT Pre-Training with Online Tokenizer"

[`arXiv`](https://arxiv.org/abs/2111.07832)

## Start

We provide `run.sh` with which you can complete the pre-training + fine-tuning experiment cycle in an one-line command.

### Training for Pretrained


### Training for Downstream Task


### Test TIPC


## Performance

You can choose to download only the weights of the pre-trained `backbone` used for downstream tasks, and the full ckpt which contains `backbone` and projection head weights for both student and teacher networks. 
For the `backbone`, `s` denotes that the student network is selected while `t` denotes that the teacher network is selected. `PS` denotes prediction shape.

### Pretrained Models

<table>
  <tr>
    <th>Arch.</th>
    <th>Par.</th>
    <th>PS</th>
    <th>k-NN</th>
    <th>Lin.</th>
    <th>Fin.</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>Block</td>
    <td>75.2%</td>
    <td>77.9%</td>
    <td>82.3%</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>Swin-T/7</td>
    <td>28M</td>
    <td>Block</td>
    <td>75.3%</td>
    <td>78.6%</td>
    <td>\</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_7/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_7/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_7/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_7/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>Swin-T/14</td>
    <td>28M</td>
    <td>Block</td>
    <td>76.2%</td>
    <td>79.3%</td>
    <td>\</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_14/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_14/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_14/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_14/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>Block</td>
    <td>77.1%</td>
    <td>79.5%</td>
    <td>84.0%</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>Rand</td>
    <td>77.3%</td>
    <td>79.8%</td>
    <td>84.1%</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>307M</td>
    <td>Block</td>
    <td>78.0%</td>
    <td>81.0%</td>
    <td>84.8%</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>307M</td>
    <td>Rand</td>
    <td>77.7%</td>
    <td>81.3%</td>
    <td>85.0%</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint_teacher.pth">backbone (t)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/log.txt">logs</a></td>
  </tr>
</table>

We also provide the ViT-{B,L}/16 model pre-trained on ImageNet-22K dataset.

 <table>
  <tr>
    <th rowspan="2">Arch.</th>
    <th rowspan="2">Par.</th>
    <th rowspan="2">PS</th>
    <th rowspan="2">k-NN</th>
    <th rowspan="2">Lin.</th>
    <th colspan="3">Fin.</th>
    <th rowspan="2" colspan="6">download</th>
  </tr>
  <tr>
  <th>256</th>
  <th>384</th>
  <th>512</th>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>Block</td>
    <td>71.1%</td>
    <td>79.0%</td>
    <td>84.4%</td>
    <td>\</td>
    <td>\</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint_student.pth">backbone (s)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>307M</td>
    <td>Block</td>
    <td>72.9%</td>
    <td>82.3%</td>
    <td>86.6%</td>
    <td>87.5%</td>
    <td>87.8%</td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint_student.pth">backbone (s)</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/args.txt">args</a></td>
    <td><a href="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/log.txt">logs</a></td>
  </tr>
</table>

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation:

```
@article{zhou2021ibot,
  title={iBOT: Image BERT Pre-Training with Online Tokenizer},
  author={Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
  journal={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```