import torch
import torch.nn as nn
from mmcv.cnn import (
        ConvModule,
        build_upsample_layer,
        build_norm_layer,
        build_activation_layer,
        UPSAMPLE_LAYERS
    )
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint
from models.basic_models import DeconvModule
from models.attention_module import Block


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 ):
        super(BasicConvBlock, self).__init__()

        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        return out


class UpConvBlock(nn.Module):
    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='DeconvModule'),
                 skip_connection=True,
                 # Attention
                 num_heads=1,
                 mlp_ratios=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratios=1,
                 depths=2,
                 number_frames=4,
                 ):
        super(UpConvBlock, self).__init__()
        self.skip_connection = skip_connection
        self.depths = depths
        # Upsample layer
        self.upsample = build_upsample_layer(
            cfg=upsample_cfg,
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # self.attention = (Block(
        #     dim=skip_channels * 2, num_heads=num_heads, mlp_ratio=mlp_ratios,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
        #     attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
        #     sr_ratio=sr_ratios)
        # )

        self.attention = nn.ModuleList([Block(
            dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratios,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            number_frames=number_frames)
            for j in range(depths)])

        self.attention2 = nn.ModuleList([Block(
            dim=skip_channels, num_heads=num_heads, mlp_ratio=mlp_ratios,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            number_frames=number_frames)
            for j in range(depths)])

        # Convolution layer
        self.conv_block = conv_block(
            in_channels=2 * skip_channels if skip_connection else skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x, f1, f2, skip):
        if self.skip_connection:
            b, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

            f2 = f2.flatten(2).transpose(1, 2)
            for i in range(self.depths):
                x = self.attention[i](x, f2, H, W)
            # x = self.attention(x, H, W)
            x = x.reshape(b, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # Upsample
        x = self.upsample(x)
        # Attention 2
        if self.skip_connection:
            f1 = f1.flatten(2).transpose(1, 2)
            b, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

            for i in range(self.depths):
                x = self.attention2[i](x, f1, H, W)
            x = x.reshape(b, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.skip_connection:
            x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)



        return x


class DSCG(nn.Module):
    def __init__(self,
                 number_frames=[4, 4, 4, 4, 4, 4],
                 embed_dim=[64, 128, 320, 512],
                 extra_stages=2,
                 num_convs=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='DeconvModule'),
                 # Attention
                 num_heads=[1, 2, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2, 2],
                 ):
        super(DSCG, self).__init__()
        self.extra_stages = extra_stages
        num_stages = len(embed_dim)         # 4
        num_stages += self.extra_stages     # 4 + 2 = 6

        extra_embed_dim = [int(embed_dim[0] * 2 ** ((-i) - 1))
                           for i in reversed(range(self.extra_stages))]

        self.embed_dim = [y for x in [extra_embed_dim, embed_dim] for y in x]

        # Upsample
        for i in range(num_stages - 1):     # Upsampling first, then convolution
            decoder = UpConvBlock(
                conv_block=BasicConvBlock,
                in_channels=self.embed_dim[i + 1] * number_frames[i+1],
                skip_channels=self.embed_dim[i] * number_frames[i],
                out_channels=self.embed_dim[i] * number_frames[i],
                num_convs=num_convs,
                stride=1,
                dilation=1,
                conv_cfg=conv_cfg,  # None
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg,
                skip_connection=True if i >= self.extra_stages else False,
                num_heads=num_heads[i],
                mlp_ratios=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                depths=depths[i],
                number_frames=number_frames[i]
            )
            deconv_name = f'deconv_{i}'
            self.add_module(deconv_name, decoder)

        # Last layer
        in_chs = self.embed_dim[0] * number_frames[0]
        self.conv = nn.Sequential(
            # ConvModule(in_channels=in_chs, out_channels=in_chs, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvBnRelu(in_chs, in_chs, kernel_size=3, padding=1),
            nn.Conv2d(in_chs, 3, kernel_size=3, padding=1, bias=False)
        )

        for i in range(num_stages - 1):
            deconv = getattr(self, f'deconv_{i}')
            initialize_weights(deconv)
        initialize_weights(self.conv)

    def forward(self, x, f1, f2):
        out = x[-1]
        stages = len(self.embed_dim) - 1
        for i in reversed(range(stages)):
            decoder = getattr(self, f'deconv_{i}')
            if i >= self.extra_stages:
                f1i = f1[i - self.extra_stages]
                f2i = f2[i - self.extra_stages+1]
                skip = x[i - self.extra_stages]
                out = decoder(out, f1i, f2i, skip)
            else:
                out = decoder(out, out, out, out)
        out = self.conv(out)

        return out
    '''
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNetDec, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    '''
    def init_weights(self, logger, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')
