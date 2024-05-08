# ------------------------------------------------------------------------
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA) and S^2(https://github.com/bfshi/scaling_on_scales)
# Copyright 2024 Jiachen Li
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import torch.nn.functional as F
from transformers.activations import ACT2FN

import math
from einops import rearrange

from .clip import CLIPVisionTransformer
from .clip_smoe import CLIPSMoEVisionTransformer

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.clip_smoe = args.clip_smoe
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.scales = args.scales
        if args.clip_smoe:
            self.vision_model = CLIPSMoEVisionTransformer(self.cfg_only, num_experts=args.num_experts, num_selected=args.num_selected)
        else:
            self.vision_model = CLIPVisionTransformer(self.cfg_only)
        self.is_loaded = True

    def feature_select(self, image_features):
        #image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def split_chessboard(self, x, num_split):
        """
            x: b * c * h * w
            Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
        """
        B, C, H, W = x.shape
        assert H % num_split == 0 and W % num_split == 0
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split) for j in range(num_split)], dim=0)
        return x_split

    def merge_chessboard(self, x, num_split):
        """
            x: b * c * h * w
            Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
            (inverse of split_chessboard)
        """
        B, C, H, W = x.shape
        assert B % (num_split**2) == 0
        b = B // (num_split**2)
        x_merge = torch.cat([torch.cat([x[(i*num_split + j)*b:(i*num_split + j + 1)*b] for j in range(num_split)], dim=-1) for i in range(num_split)], dim=-2)
        return x_merge 
    
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_model(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            input_size = images.shape[3]
            img_sizes = [int(input_size * scale) for scale in self.scales]
            num_splits = [math.ceil(size / input_size) for size in img_sizes]
            image_pyramids = [images]
            for i, (size, num_split) in enumerate(zip(img_sizes, num_splits)):
                if i > 0:
                    x = F.interpolate(images.to(torch.float32), size=size, mode='bicubic').to(images.dtype)
                    x = self.split_chessboard(x, num_split=num_split)
                    image_pyramids.append(x)
            if self.clip_smoe:
                image_features = []
                balance_losses = []
                router_z_losses = []
                for i, (x, num_split) in enumerate(zip(image_pyramids, num_splits)):
                    out_x, balance_loss, router_z_loss = self.vision_model(x)
                    out_x = self.feature_select(out_x)
                    if i > 0:
                        out_x = rearrange(out_x, 'b (h w) c -> b c h w', h=int(out_x.shape[1] ** 0.5), w=int(out_x.shape[1] ** 0.5))
                        out_x = self.merge_chessboard(out_x, num_split=num_split)
                        out_x = F.interpolate(out_x.to(torch.float32), size=int(image_features[0].shape[1] ** 0.5), mode='area').to(x.dtype)
                        out_x = rearrange(out_x, 'b c h w -> b (h w) c')
                    image_features.append(out_x)
                    balance_losses.append(balance_loss)
                    router_z_losses.append(router_z_loss)
                image_features = torch.cat(image_features, dim=-1)
                return image_features, torch.stack(balance_losses).mean(), torch.stack(router_z_losses).mean()
            else:
                image_features = []
                for i, (x, num_split) in enumerate(zip(image_pyramids, num_splits)):
                    out_x = self.vision_model(x)
                    out_x = self.feature_select(out_x)
                    if i > 0:
                        out_x = rearrange(out_x, 'b (h w) c -> b c h w', h=int(out_x.shape[1] ** 0.5), w=int(out_x.shape[1] ** 0.5))
                        out_x = self.merge_chessboard(out_x, num_split=num_split)
                        out_x = F.interpolate(out_x.to(torch.float32), size=int(image_features[0].shape[1] ** 0.5), mode='area').to(x.dtype)
                        out_x = rearrange(out_x, 'b c h w -> b (h w) c')
                    image_features.append(out_x)
                image_features = torch.cat(image_features, dim=-1)
                return image_features, None, None

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def device(self):
        return self.vision_model.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_model.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
