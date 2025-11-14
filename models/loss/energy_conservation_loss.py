import torch
import torch.nn as nn
import torch.nn.functional as F

class energy_conservation_loss(nn.Module):

    def __init__(self,
                 init_k=0.5,
                 init_C=1.0,
                 delta_t=0.1,
                 stefan_boltzmann=5.67e-8,
                 spectral_bandwidth=6e-6,
                 epsilon_init=0.95):
        super().__init__()

        self.register_buffer('stefan', torch.tensor(stefan_boltzmann))
        self.register_buffer('bandwidth', torch.tensor(spectral_bandwidth))

        self.epsilon = nn.Parameter(torch.tensor(epsilon_init))
        self.k = nn.Parameter(torch.tensor(init_k))
        self.C = nn.Parameter(torch.tensor(init_C))
        self.delta_t = nn.Parameter(torch.tensor(delta_t))
        self.alpha = nn.Parameter(torch.tensor(0.2))

        self.register_buffer('laplacian_gradient', torch.tensor(
            [[[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]]], dtype=torch.float32))

        self.register_buffer('laplacian_kernel', torch.tensor(
            [[[[0.5, 1.0, 0.5],
               [1.0, -6.0, 1.0],
               [0.5, 1.0, 0.5]]]], dtype=torch.float32))

    def _compute_grad(self, x):

        output = F.conv2d(x, self.laplacian_gradient, padding=1)
        output = torch.abs(output)
        return output

    def _intensity_to_temperature(self, x, visible=None):

        epsilon_base = torch.sigmoid(self.epsilon) * 0.3 + 0.69  # [0.69, 0.99]

        if visible is not None:
            epsilon = epsilon_base - torch.sigmoid(self.alpha) * 0.3 * visible
            epsilon = torch.clamp(epsilon, 0.69, 0.99)
        else:
            epsilon = epsilon_base

        return (x / (epsilon * self.stefan * self.bandwidth + 1e-6)).pow(0.25)

    def _energy_conservation(self, T_fused, T_ir, visible_grad):

        k_map = self.k.abs() * (1 - visible_grad)

        delta_T = T_fused - T_ir
        laplacian_ir = F.conv2d(T_ir, self.laplacian_kernel, padding=1)
        energy_diff = self.C.abs() * delta_T - k_map * laplacian_ir * self.delta_t.abs()

        weight = torch.sigmoid(10 * delta_T.abs() - 0.5)
        return torch.mean(weight * energy_diff.abs())

    def forward(self, fused, infrared, visible):

        T_fused = torch.sigmoid(fused)
        T_ir = torch.sigmoid(infrared)
        visible_norm = torch.sigmoid(visible)


        T_fused = self._intensity_to_temperature(T_fused, visible_norm)
        T_ir = self._intensity_to_temperature(T_ir)

        visible_grad = self._compute_grad(visible_norm)
        visible_grad = visible_grad / (visible_grad.max() + 1e-6)

        energy_loss = self._energy_conservation(T_fused, T_ir, visible_grad)

        return 2 * energy_loss