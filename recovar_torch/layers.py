import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

BN_MOMENTUM = 0.01
BN_EPS = 1e-3


class AddNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        return x + torch.randn_like(x) * self.stddev


class NormalizeStd(nn.Module):
    def __init__(self, axis=1, eps=1e-27):
        super().__init__()
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        std = torch.std(x, dim=self.axis, keepdim=True, unbiased=False)
        return x / (self.eps + std)


class Padding(nn.Module):
    def __init__(self, size=(0, 0)):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.pad(x, (self.size[0], self.size[1]), mode="reflect")


class Conv(nn.Module):
    def __init__(self, in_channels, num_of_filters, filter_kernel_size):
        super().__init__()
        self.padding = Padding((filter_kernel_size // 2, filter_kernel_size // 2))
        self.conv = nn.Conv1d(in_channels, num_of_filters, filter_kernel_size)
        self.bn = nn.BatchNorm1d(num_of_filters, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return torch.transpose(x, 1, 2)


class Downsample(nn.Module):
    def __init__(self, in_channels, num_of_filters, filter_kernel_size):
        super().__init__()
        self.padding = Padding((filter_kernel_size // 2, filter_kernel_size // 2))
        self.conv = nn.Conv1d(in_channels, num_of_filters, filter_kernel_size, stride=2)
        self.bn = nn.BatchNorm1d(num_of_filters, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return torch.transpose(x, 1, 2)


class Upsample(nn.Module):
    def __init__(self, in_channels, num_of_filters, filter_kernel_size):
        super().__init__()
        self.padding = Padding((filter_kernel_size // 2, filter_kernel_size // 2))
        self.conv = nn.Conv1d(in_channels, num_of_filters, filter_kernel_size)
        self.bn = nn.BatchNorm1d(num_of_filters, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.repeat_interleave(x, 2, dim=2)
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return torch.transpose(x, 1, 2)


class UpsampleNoactivation(nn.Module):
    def __init__(self, in_channels, num_of_filters, filter_kernel_size):
        super().__init__()
        self.padding = Padding((filter_kernel_size // 2, filter_kernel_size // 2))
        self.conv = nn.Conv1d(in_channels, num_of_filters, filter_kernel_size)
        self.bn = nn.BatchNorm1d(num_of_filters, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.repeat_interleave(x, 2, dim=2)
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        return torch.transpose(x, 1, 2)


class ResIdentity(nn.Module):
    def __init__(self, in_channels, num_of_filters, filter_kernel_size):
        super().__init__()
        self.padding = Padding((filter_kernel_size // 2, filter_kernel_size // 2))
        self.conv1 = nn.Conv1d(in_channels, num_of_filters, filter_kernel_size)
        self.bn1 = nn.BatchNorm1d(num_of_filters, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv1d(num_of_filters, num_of_filters, filter_kernel_size)
        self.bn2 = nn.BatchNorm1d(num_of_filters, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x_skip = x

        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.padding(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x + x_skip
        x = F.relu(x)
        return torch.transpose(x, 1, 2)


class CrossCovarianceCircular(nn.Module):
    def forward(self, a, b):
        timesteps = a.shape[1]
        channels = a.shape[2]

        a = a - torch.mean(a, dim=1, keepdim=True)
        b = b - torch.mean(b, dim=1, keepdim=True)

        a = a / torch.sqrt(torch.tensor(float(channels)))
        b = b / torch.sqrt(torch.tensor(float(channels)))

        a = torch.transpose(a, 1, 2)
        b = torch.transpose(b, 1, 2)

        a = a.to(torch.complex128)
        b = b.to(torch.complex128)

        aw = torch.fft.fft(a)
        bw = torch.fft.fft(b)

        cw = aw * torch.conj(bw)

        c = torch.fft.ifft(cw)

        c = torch.real(c)
        c = c.to(torch.float32)
        c = torch.roll(c, shifts=timesteps // 2, dims=2)

        c = torch.sum(c, dim=1)
        return c

    @staticmethod
    def _fftw(n, d=1):
        f1 = torch.arange(0, (n + 1) // 2) / (n * d)
        f2 = torch.arange(-(n - 1) // 2, 0) / (n * d)
        f = torch.cat([f1, f2], dim=0)
        w = (2 * pi) * f
        return w.to(torch.complex128)
