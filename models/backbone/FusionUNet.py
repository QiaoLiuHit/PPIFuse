import torch
import torch.nn as nn
from .swin_transformer import BasicLayer, Cross_BasicLayer
from .ThermalDiffusionConv import ThermalDiffusionConv
from .StefanThermalAtten import StefanThermalAtten

def add_conv_stage(depth, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    if depth == 1:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )
    elif depth == 2:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Tanh(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )
    elif depth == 3:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Tanh(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )


def channel_tune(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    )


def add_ThermalDiffusionConv(in_chans=1, out_chans=16, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        ThermalDiffusionConv(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
    )


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


def new_recon(img_tensor, input_size, out_channels, patch_size=1):
    B, num_patches, patch_dim = img_tensor.shape
    output = img_tensor.view(B, num_patches, out_channels, patch_size, patch_size)
    # output = output.view(B,int(num_patches ** 0.5),int(num_patches ** 0.5),out_channels,patch_size,patch_size)
    output = output.view(B, input_size[0], input_size[1], out_channels, patch_size, patch_size)
    output = output.permute(0, 3, 1, 4, 2, 5)
    output = output.contiguous().view(B, out_channels, input_size[0], input_size[1])
    return output


class ir_extract_fea(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ir_extract_fea, self).__init__()
        self.ThermalDiffusionConv = ThermalDiffusionConv(dim_in, dim_out)

    def forward(self, x):
        x1 = self.ThermalDiffusionConv(x)
        return x1


class interaction(nn.Module):
    def __init__(self, in_chans, out_chans, input_resolution, depth, num_heads, windos_size,
                 mlp_ratio, drop, attn_drop, drop_path, qkv_bias=True, qk_scale=None):
        super(interaction, self).__init__()
        self.out_chans = out_chans
        self.CrossAtten = Cross_BasicLayer(dim=in_chans,
                                           input_resolution=input_resolution,
                                           depth=depth,
                                           num_heads=num_heads,
                                           window_size=windos_size,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path)
        self.standard_Conv = channel_tune(in_chans * 2, out_chans)

    def forward(self, ir, vis, input_size):
        self.CrossAtten.input_resolution = input_size
        vis = vis.flatten(2).transpose(1, 2)
        ir = ir.flatten(2).transpose(1, 2)
        ir_vis_a, ir_vis_b = self.CrossAtten(vis, ir, input_size)
        ir_vis_a = new_recon(ir_vis_a, input_size, self.out_chans)
        ir_vis_b = new_recon(ir_vis_b, input_size, self.out_chans)
        ir_vis = torch.cat((ir_vis_a, ir_vis_b), dim=1)
        ir_vis = self.standard_Conv(ir_vis)
        return ir_vis


# fusion network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.max_pool = nn.MaxPool2d(2)

        self.init_conv_vis = channel_tune(1, 16, 1, 1, 0)

        self.ir_extract_fea1 = ir_extract_fea(1, 16)
        self.ir_extract_fea2 = ir_extract_fea(16, 32)
        self.ir_extract_fea3 = ir_extract_fea(32, 64)
        self.ir_extract_fea4 = ir_extract_fea(64, 128)

        self.StefanAtten1 = StefanThermalAtten(16, 4)
        self.StefanAtten2 = StefanThermalAtten(32, 4)
        self.StefanAtten3 = StefanThermalAtten(64, 4)
        self.StefanAtten4 = StefanThermalAtten(128, 4)

        self.dpr = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.interaction1 = interaction(in_chans=16, out_chans=16,
                                        input_resolution=(256, 256),
                                        depth=2, num_heads=1, windos_size=8,
                                        mlp_ratio=4., drop=0., attn_drop=0.,
                                        drop_path=self.dpr[0:2])

        self.interaction2 = interaction(in_chans=32, out_chans=32,
                                        input_resolution=(128, 128),
                                        depth=2, num_heads=1, windos_size=8,
                                        mlp_ratio=4., drop=0.1, attn_drop=0.1,
                                        drop_path=self.dpr[2:4])

        self.interaction3 = interaction(in_chans=64, out_chans=64,
                                        input_resolution=(64, 64),
                                        depth=2, num_heads=1, windos_size=8,
                                        mlp_ratio=4., drop=0., attn_drop=0.,
                                        drop_path=self.dpr[4:6])

        self.interaction4 = interaction(in_chans=128, out_chans=128,
                                        input_resolution=(32, 32),
                                        depth=2, num_heads=2, windos_size=8,
                                        mlp_ratio=4., drop=0.1, attn_drop=0.1,
                                        drop_path=self.dpr[6:8])

        self.add_channel1 = channel_tune(16, 32, 1, 1, 0)
        self.add_channel2 = channel_tune(32, 64, 1, 1, 0)
        self.add_channel3 = channel_tune(64, 128, 1, 1, 0)

        self.basic_layer1 = BasicLayer(dim=16, input_resolution=(256, 256),
                                       depth=2, num_heads=1, window_size=8,
                                       mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                       drop=0., attn_drop=0., drop_path=self.dpr[0:2])
        self.basic_layer2 = BasicLayer(dim=32, input_resolution=(128, 128),
                                       depth=2, num_heads=1, window_size=8,
                                       mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                       drop=0.1, attn_drop=0.1, drop_path=self.dpr[2:4])
        self.basic_layer3 = BasicLayer(dim=64, input_resolution=(64, 64),
                                       depth=2, num_heads=1, window_size=8,
                                       mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                       drop=0., attn_drop=0., drop_path=self.dpr[4:6])
        self.basic_layer4 = BasicLayer(dim=128, input_resolution=(32, 32),
                                       depth=2, num_heads=2, window_size=8,
                                       mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                       drop=0.1, attn_drop=0.1, drop_path=self.dpr[6:8])

        self.upsample1 = upsample(128, 64)
        self.upsample2 = upsample(64, 32)
        self.upsample3 = upsample(32, 16)

        self.conv_out1 = add_conv_stage(2, 128, 64)
        self.conv_out2 = add_conv_stage(2, 64, 32)
        self.conv_out3 = add_conv_stage(2, 32, 16)
        self.conv0m = nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1), nn.Tanh())


    def forward(self, img_ir, img_vis):

        img_vis = self.init_conv_vis(img_vis)

        img_ir1 = self.ir_extract_fea1(img_ir)
        input_size1 = img_ir.shape[-2:]
        img_vis1 = img_vis.flatten(2).transpose(1, 2)
        self.basic_layer1.input_resolution = input_size1
        img_vis1_out = self.basic_layer1(img_vis1, input_size1)
        img_vis1_out = new_recon(img_vis1_out, input_size1, 16)
        img_ir1 = self.StefanAtten1(img_ir1)
        ir_vis1 = self.interaction1(img_ir1,img_vis1_out,input_size1)
        img_vis1_out = img_vis1_out + ir_vis1
        ir_vis1 = img_vis1_out

        input_size2 = tuple(x // 2 for x in input_size1)
        img_ir1 = self.max_pool(img_ir1)
        img_ir2 = self.ir_extract_fea2(img_ir1)
        img_vis1_out = self.max_pool(img_vis1_out)
        img_vis1_out = self.add_channel1(img_vis1_out)
        img_vis1_out = img_vis1_out.flatten(2).transpose(1, 2)
        self.basic_layer2.input_resolution = input_size2
        img_vis2_out = self.basic_layer2(img_vis1_out, input_size2)
        img_vis2_out = new_recon(img_vis2_out, input_size2, 32)
        img_ir2 = self.StefanAtten2(img_ir2)
        ir_vis2 = self.interaction2(img_ir2, img_vis2_out, input_size2)
        img_vis2_out = img_vis2_out + ir_vis2
        ir_vis2 = img_vis2_out

        input_size3 = tuple(x // 2 for x in input_size2)
        img_ir2 = self.max_pool(img_ir2)
        img_ir3 = self.ir_extract_fea3(img_ir2)
        img_vis2_out = self.max_pool(img_vis2_out)
        img_vis2_out = self.add_channel2(img_vis2_out)
        img_vis2_out = img_vis2_out.flatten(2).transpose(1, 2)
        self.basic_layer3.input_resolution = input_size3
        img_vis3_out = self.basic_layer3(img_vis2_out, input_size3)
        img_vis3_out = new_recon(img_vis3_out, input_size3, 64)
        img_ir3 = self.StefanAtten3(img_ir3)
        ir_vis3 = self.interaction3(img_ir3, img_vis3_out, input_size3)
        img_vis3_out = img_vis3_out + ir_vis3
        ir_vis3 = img_vis3_out

        input_size4 = tuple(x // 2 for x in input_size3)
        img_ir3 = self.max_pool(img_ir3)
        img_ir4 = self.ir_extract_fea4(img_ir3)
        img_vis3_out = self.max_pool(img_vis3_out)
        img_vis3_out = self.add_channel3(img_vis3_out)
        img_vis3_out = img_vis3_out.flatten(2).transpose(1, 2)
        self.basic_layer4.input_resolution = input_size4
        img_vis4_out = self.basic_layer4(img_vis3_out, input_size4)
        img_vis4_out = new_recon(img_vis4_out, input_size4, 128)
        img_ir4 = self.StefanAtten4(img_ir4)
        ir_vis4 = self.interaction4(img_ir4, img_vis4_out, input_size4)
        img_vis4_out = img_vis4_out + ir_vis4
        ir_vis4 = img_vis4_out

        output1 = self.upsample1(ir_vis4)
        output1 = torch.cat((output1, ir_vis3), dim=1)
        output1 = self.conv_out1(output1)

        output2 = self.upsample2(output1)
        output2 = torch.cat((output2, ir_vis2), dim=1)
        output2 = self.conv_out2(output2)

        output3 = self.upsample3(output2)
        output3 = torch.cat((output3, ir_vis1), dim=1)
        output3 = self.conv_out3(output3)

        output = self.conv0m(output3)

        return output