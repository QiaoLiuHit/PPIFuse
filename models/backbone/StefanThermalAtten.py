import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePowerScale(nn.Module):

    def __init__(self, init_exponent=4.0, init_scale=1.0):
        super().__init__()
        self.log_exponent = nn.Parameter(torch.log(torch.tensor(init_exponent)))
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        exponent = torch.exp(self.log_exponent)
        return x ** exponent * self.scale

class StefanThermalAtten(nn.Module):

    def __init__(self, in_channels, reduction=16, T_scale=1.0):
        super().__init__()

        mid_temp = max(1, in_channels // reduction)
        self.temp_estimator = nn.Sequential(
            nn.Conv2d(in_channels, mid_temp * reduction * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_temp * reduction * 2, mid_temp * reduction, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_temp * reduction, mid_temp, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_temp, 1, 1),
            nn.Sigmoid(),
            LearnablePowerScale(init_exponent=4.0, init_scale=T_scale)
        )

        mid_emis = max(4, in_channels // 4)
        self.emis_net = nn.Sequential(
            nn.Conv2d(in_channels, mid_emis * 4 * 2, 3, padding=1),
            nn.GroupNorm(4, mid_emis * 4 * 2),
            nn.GELU(),
            nn.Conv2d(mid_emis * 4 * 2, mid_emis * 4, 3, padding=1),
            nn.GroupNorm(4, mid_emis * 4),
            nn.GELU(),
            nn.Conv2d(mid_emis * 4, mid_emis, 3, padding=1),
            nn.GroupNorm(4, mid_emis),
            nn.GELU(),
            nn.Conv2d(mid_emis, 1, 1),
            nn.Hardtanh(min_val=0.01, max_val=0.99)
        )

        self.phys_fusion = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.grad_conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self._init_grad_conv()

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # 注意力融合系数
        self.alpha = nn.Parameter(torch.tensor([0.0]))

    def _init_grad_conv(self):

        self.grad_conv.weight.data[0, 0] = torch.tensor(
            [[[[1.0, 0.0, -1.0],
               [2.0, 0.0, -2.0],
               [1.0, 0.0, -1.0]]]],
            dtype=torch.float32
        )

        self.grad_conv.weight.data[1, 0] = torch.tensor(
            [[[[1.0, 2.0, 1.0],
               [0.0, 0.0, 0.0],
               [-1.0, -2.0, -1.0]]]],
            dtype=torch.float32
        )

    def _spatial_features(self, x):

        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        x_min = x.amin(dim=1, keepdim=True)


        grad = self.grad_conv(x_mean)
        grad_x, grad_y = grad.chunk(2, dim=1)
        grad_x = grad_x.abs()
        grad_y = grad_y.abs()

        return torch.cat([x_mean, x_std, x_max - x_min, grad_x, grad_y], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape

        T_map = self.temp_estimator(x)  # [B,1,H,W]

        ε_map = self.emis_net(x)  # [B,1,H,W]

        phys_feat = torch.cat([T_map, ε_map], dim=1)
        att_phys = self.phys_fusion(phys_feat)

        spatial_feat = self._spatial_features(x)
        att_spatial = self.spatial_conv(spatial_feat)

        alpha = torch.sigmoid(self.alpha)
        att_weight = alpha * att_phys + (1 - alpha) * att_spatial

        return x * (1 + att_weight)

