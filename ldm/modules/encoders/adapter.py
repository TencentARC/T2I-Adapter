import torch
import torch.nn as nn
from collections import OrderedDict
from ldm.modules.extra_condition.api import ExtraCondition
from ldm.modules.diffusionmodules.util import zero_module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        super(Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(nn.Module):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()

        scale = width ** -0.5
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        # x shape [N, HW+1, C]
        style_embedding = self.style_embedding + torch.zeros(
            (x.shape[0], self.num_token, self.style_embedding.shape[-1]), device=x.device)
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token:, :])
        x = x @ self.proj

        return x


class ResnetBlock_light(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        return h + x


class extractor(nn.Module):
    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)

        return x


class Adapter_light(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64):
        super(Adapter_light, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            if i == 0:
                self.body.append(extractor(in_c=cin, inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                self.body.append(extractor(in_c=channels[i-1], inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=True))
        self.body = nn.ModuleList(self.body)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(x)

        return features


class CoAdapterFuser(nn.Module):
    def __init__(self, unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3):
        super(CoAdapterFuser, self).__init__()
        scale = width ** 0.5
        # 16, maybe large enough for the number of adapters?
        self.task_embedding = nn.Parameter(scale * torch.randn(16, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(len(unet_channels), width))
        self.spatial_feat_mapping = nn.ModuleList()
        for ch in unet_channels:
            self.spatial_feat_mapping.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(ch, width),
            ))
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.spatial_ch_projs = nn.ModuleList()
        for ch in unet_channels:
            self.spatial_ch_projs.append(zero_module(nn.Linear(width, ch)))
        self.seq_proj = nn.Parameter(torch.zeros(width, width))

    def forward(self, features):
        if len(features) == 0:
            return None, None
        inputs = []
        for cond in features.keys():
            task_idx = getattr(ExtraCondition, cond.name).value
            if not isinstance(features[cond], list):
                inputs.append(features[cond] + self.task_embedding[task_idx])
                continue

            feat_seq = []
            for idx, feature_map in enumerate(features[cond]):
                feature_vec = torch.mean(feature_map, dim=(2, 3))
                feature_vec = self.spatial_feat_mapping[idx](feature_vec)
                feat_seq.append(feature_vec)
            feat_seq = torch.stack(feat_seq, dim=1)  # Nx4xC
            feat_seq = feat_seq + self.task_embedding[task_idx]
            feat_seq = feat_seq + self.positional_embedding
            inputs.append(feat_seq)

        x = torch.cat(inputs, dim=1)  # NxLxC
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        ret_feat_map = None
        ret_feat_seq = None
        cur_seq_idx = 0
        for cond in features.keys():
            if not isinstance(features[cond], list):
                length = features[cond].size(1)
                transformed_feature = features[cond] * ((x[:, cur_seq_idx:cur_seq_idx+length] @ self.seq_proj) + 1)
                if ret_feat_seq is None:
                    ret_feat_seq = transformed_feature
                else:
                    ret_feat_seq = torch.cat([ret_feat_seq, transformed_feature], dim=1)
                cur_seq_idx += length
                continue

            length = len(features[cond])
            transformed_feature_list = []
            for idx in range(length):
                alpha = self.spatial_ch_projs[idx](x[:, cur_seq_idx+idx])
                alpha = alpha.unsqueeze(-1).unsqueeze(-1) + 1
                transformed_feature_list.append(features[cond][idx] * alpha)
            if ret_feat_map is None:
                ret_feat_map = transformed_feature_list
            else:
                ret_feat_map = list(map(lambda x, y: x + y, ret_feat_map, transformed_feature_list))
            cur_seq_idx += length

        assert cur_seq_idx == x.size(1)

        return ret_feat_map, ret_feat_seq
