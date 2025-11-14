import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalDiffusionConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()

        self.diffusion_net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 8 * out_channels, 1)
        )

        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        b, c_in, h, w = x.shape


        diffusion_map = torch.sigmoid(self.diffusion_net(x)).view(b, self.out_channels, 8)

        D_h1 = diffusion_map[:, :, 0]
        D_h2 = diffusion_map[:, :, 1]
        D_v1 = diffusion_map[:, :, 2]
        D_v2 = diffusion_map[:, :, 3]
        D_d1 = diffusion_map[:, :, 4]
        D_d2 = diffusion_map[:, :, 5]
        D_d3 = diffusion_map[:, :, 6]
        D_d4 = diffusion_map[:, :, 7]

        kernel = self._build_thermal_kernel(D_h1, D_h2, D_v1, D_v2, D_d1, D_d2, D_d3, D_d4)

        x_reshaped = x.view(1, b * c_in, h, w)
        kernel = kernel.unsqueeze(2).expand(-1, -1, c_in, -1, -1)
        kernel_reshaped = kernel.reshape(b * self.out_channels, c_in,
                                       self.kernel_size, self.kernel_size)
        output = F.conv2d(
            x_reshaped,
            kernel_reshaped,
            stride=self.stride,
            padding=self.padding,
            groups=b
        )
        output = output.view(b, self.out_channels, h, w)
        output = self.relu(output)
        return output + self.base_conv(x)

    def _build_thermal_kernel(self, D_h1, D_h2, D_v1, D_v2, D_d1, D_d2, D_d3, D_d4):

        device = D_h1.device
        b, c_out = D_h1.shape
        kernel = torch.zeros((b, c_out, self.kernel_size, self.kernel_size), device=device)
        center = self.kernel_size // 2

        kernel[:, :, center, center] = -1 * (D_h1 + D_h2 + D_v1 + D_v2)

        kernel[:, :, center, center+1] = D_h1
        kernel[:, :, center, center-1] = D_h2
        kernel[:, :, center+1, center] = D_v1
        kernel[:, :, center-1, center] = D_v2

        kernel[:, :, center-1, center-1] = D_d1
        kernel[:, :, center-1, center+1] = D_d2
        kernel[:, :, center+1, center+1] = D_d3
        kernel[:, :, center+1, center-1] = D_d4

        return kernel