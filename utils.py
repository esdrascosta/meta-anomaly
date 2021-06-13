import torch
import torch.nn.functional as F
from torch import Tensor

def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)
