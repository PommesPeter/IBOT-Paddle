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

import numpy as np

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, Normal, TruncatedNormal

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class DINOHead(nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm=None,
        act_layer=nn.GELU,
        last_norm=None,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
        epsilon=1e-5,
        **kwargs
    ):
        super().__init__()
        if norm is not None:
            self.norm = eval(norm)(hidden_dim, epsilon=epsilon)
        if last_norm is not None:
            self.last_norm = eval(last_norm)(out_dim, epsilon=epsilon)
        else:
            self.last_norm = None
        if act_layer is not None:
            self.act = act_layer()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(self.act)

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(self.act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, out_dim, bias_attr=False), dim=1
            )
            ones_(self.last_layer.weight_g)
            if norm_last_layer:
                self.last_layer.weight_g.stop_gradient = False

        else:
            self.last_layer = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, axis=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x


class IBOTHead(DINOHead):
    def __init__(
        self,
        *args,
        patch_out_dim=8192,
        norm=None,
        act_layer=nn.GELU,
        last_norm=None,
        nlayers=3,
        epsilon=1e-5,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
        shared_head=False,
        **kwargs
    ):
        super(IBOTHead, self).__init__(
            *args,
            norm=norm,
            act_layer=act_layer,
            last_norm=last_norm,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            norm_last_layer=norm_last_layer,
            **kwargs
        )
        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(
                    nn.Linear(bottleneck_dim, patch_out_dim, bias_attr=False), dim=1
                )
                ones_(self.last_layer2.weight_g)
                if norm_last_layer:
                    self.last_layer2.weight_g.stop_gradient = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            if last_norm is not None:
                self.last_norm2 = eval(last_norm)(patch_out_dim, epsilon=epsilon)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None
            if last_norm is not None:
                self.last_norm2 = self.last_norm

    def forward(self, x):

        if len(x.shape) == 2:
            return super(IBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, axis=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])

        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)

        return x1, x2
