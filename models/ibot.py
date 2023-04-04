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

import math
from collections.abc import Callable

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, Normal, TruncatedNormal

from .utils import (load_dygraph_pretrain,
                                 load_dygraph_pretrain_from_url)
from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer
from .head import IBOTHead

MODEL_URLS = {
    "IBOT_ViT_small_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
    "IBOT_ViT_base_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
    "IBOT_ViT_large_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams",
    "IBOT_Swin_tiny_patch7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams",
    "IBOT_Swin_tiny_patch14_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
}

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class MultiCropWrapper(nn.Layer):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = paddle.cumsum(
            paddle.unique_consecutive(
                paddle.to_tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx, output = 0, paddle.empty((0,))
        for end_idx in idx_crops:
            inp_x = paddle.concat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = paddle.concat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = paddle.concat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


class IBOTVisionTransformer(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        class_num=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer="nn.LayerNorm",
        epsilon=1e-5,
        return_all_tokens=False,
        masked_im_modeling=False,
        **kwargs
    ):
        super(IBOTVisionTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            class_num,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            epsilon,
            **kwargs
        )
        self.return_all_tokens = return_all_tokens
        self.masked_im_modeling = masked_im_modeling
        self.img_size = img_size
        self.patch_size = patch_size
        # self.head = IBOTHead(in_dim=embed_dim,out_dim=out_dim,norm=norm,)

        if self.masked_im_modeling:
            self.masked_embed = self.create_parameter(
                shape=[1, embed_dim], default_initializer=zeros_
            )
        # trunc_normal_(self.masked_embed)
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return paddle.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)

    def forward_features(self, x, mask=None, return_all_tokens=None):
        # B = x.shape[0]
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        C,N,HW = x.shape
        H,W = int(self.img_size/self.patch_size),int(self.img_size/self.patch_size)
        x = x.reshape([C,N,H,W])
        # mask image modeling
        if self.masked_im_modeling:
            assert mask is not None
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(perm=[0, 2, 1])

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand((B, -1, -1)).astype(x.dtype)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # if self.fc_norm is not None:
        #     x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = (
            self.return_all_tokens if return_all_tokens is None else return_all_tokens
        )

        if return_all_tokens:
            return x

        return x[:, 0]

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask, return_all_tokens=self.return_all_tokens)
        # x = self.head(x)

        return x

    def mask_model(self, x, mask):
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        x = paddle.where(mask.unsqueeze(-1), paddle.cast(self.masked_embed, x.dtype), x)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        return x


class IBOTSwinTransformer(SwinTransformer):
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        class_num=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        pretrained_window_sizes=[0, 0, 0, 0],
        return_all_tokens=False,
        masked_im_modeling=False,
        **kwargs
    ):
        super(IBOTSwinTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            class_num,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            pretrained_window_sizes,
            **kwargs
        )
        self.return_all_token = return_all_tokens
        self.masked_im_modeling = masked_im_modeling

        if self.masked_im_modeling:
            self.masked_embed = self.create_parameter(
                shape=[1, embed_dim], default_initializer=zeros_
            )

        # self.head = IBOTHead()

    def forward_features(self, x, mask=None, return_all_tokens=None):
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(perm=[0, 2, 1])

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x_region = self.norm(x)  # B L C
        x = self.avgpool(x_region.transpose([0, 2, 1]))  # B C 1
        x = paddle.flatten(x, 1)

        return_all_tokens = (
            self.return_all_tokens if return_all_tokens is None else return_all_tokens
        )
        if return_all_tokens:
            return paddle.concat([x.unsqueeze(1), x_region], axis=1)

        return x

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask, self.return_all_tokens)
        # x = self.head(x)
        return x

    def mask_model(self, x, mask):
        # extend mask for hierarchical features
        if x.shape[-2:] != mask.shape[-2:]:
            htimes, wtimes = np.array(x.shape[-2:]) // np.array(mask.shape[-2:])
            mask = mask.repeat_interleave(htimes, -2).repeat_interleave(wtimes, -1)

        # mask embed
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)

        return x


def _load_pretrained(
    pretrained,
    model,
    model_url,
    use_ssld=False,
    use_imagenet22k_pretrained=False,
    use_imagenet22kto1k_pretrained=False,
):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(
            model,
            model_url,
            use_ssld=use_ssld,
            use_imagenet22k_pretrained=use_imagenet22k_pretrained,
            use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained,
        )
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def IBOT_ViT_small_patch16_224(pretrained=False, use_ssld=False,in_dim=384,out_dim=8192,patch_out_dim=8192,norm=None,act_layer=nn.GELU,norm_last_layer=False,shared_head=True,masked_im_modeling=False, **kwargs):
    backbone = IBOTVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qk_scale=(384 // 6) ** -0.5,
        return_all_tokens=True,
        masked_im_modeling=masked_im_modeling,
        **kwargs
    )
    _load_pretrained(
        pretrained, backbone, MODEL_URLS["IBOT_ViT_small_patch16_224"], use_ssld=use_ssld
    )
    model = MultiCropWrapper(
        backbone,
        IBOTHead(in_dim=in_dim,out_dim=out_dim,patch_out_dim=patch_out_dim,norm=norm,act_layer=act_layer,norm_last_layer=norm_last_layer,shared_head=shared_head)
    )
    return model


def IBOT_ViT_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    backbone = IBOTVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qk_scale=(768 // 12) ** -0.5,
        **kwargs
    )
    _load_pretrained(
        pretrained, backbone, MODEL_URLS["IBOT_ViT_base_patch16_224"], use_ssld=use_ssld
    )
    model = MultiCropWrapper(
        backbone,
        IBOTHead(in_dim=768, out_dim=8192, patch_out_dim=8192, norm=None, act_layer=nn.GELU, norm_last_layer=False,
                 shared_head=True)
    )
    return model


def IBOT_ViT_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    backbone = IBOTVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qk_scale=(1024 // 12) ** -0.5,
        **kwargs
    )
    _load_pretrained(
        pretrained, backbone, MODEL_URLS["IBOT_ViT_large_patch16_224"], use_ssld=use_ssld
    )
    model = MultiCropWrapper(
        backbone,
        IBOTHead(in_dim=1024, out_dim=8192, patch_out_dim=8192, norm=None, act_layer=nn.GELU, norm_last_layer=False,
                 shared_head=True)
    )
    return model


def IBOT_Swin_tiny_windows7_224(pretrained=False, use_ssld=False, **kwargs):
    backbone = IBOTSwinTransformer(
        img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        **kwargs
    )
    _load_pretrained(
        pretrained, backbone, MODEL_URLS["IBOT_Swin_tiny_windows7_224"], use_ssld=use_ssld
    )
    model = MultiCropWrapper(
        backbone,
        IBOTHead(in_dim=96, out_dim=8192, patch_out_dim=8192, norm=None, act_layer=nn.GELU, norm_last_layer=False,
                 shared_head=True)
    )
    return model

def IBOT_Swin_tiny_windows14_224(pretrained=False, use_ssld=False, **kwargs):
    backbone = IBOTSwinTransformer(
        img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=14,
        mlp_ratio=4,
        **kwargs
    )
    _load_pretrained(
        pretrained, backbone, MODEL_URLS["IBOT_Swin_tiny_windows7_224"], use_ssld=use_ssld
    )
    model = MultiCropWrapper(
        backbone,
        IBOTHead(in_dim=96, out_dim=8192, patch_out_dim=8192, norm=None, act_layer=nn.GELU, norm_last_layer=False,
                 shared_head=True)
    )
    return model
