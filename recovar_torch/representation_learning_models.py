import torch
import torch.nn as nn
from recovar_torch.config import BATCH_SIZE
from recovar_torch.layers import (
    AddNoise,
    NormalizeStd,
    Padding,
    Downsample,
    Upsample,
    UpsampleNoactivation,
    ResIdentity,
    BN_MOMENTUM,
    BN_EPS,
)
from recovar_torch.utils import demean, l2_normalize, l2_distance, l4_normalize

N_TIMESTEPS = 3000
N_CHANNELS = 3


class AutoencoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Downsample(3, 8, 15)
        self.down2 = Downsample(8, 16, 13)
        self.pad1 = Padding((1, 1))
        self.down3 = Downsample(16, 32, 11)
        self.down4 = Downsample(32, 64, 9)
        self.down5 = Downsample(64, 64, 7)

        self.resid1 = ResIdentity(64, 64, 5)
        self.resid2 = ResIdentity(64, 64, 5)
        self.resid3 = ResIdentity(64, 64, 5)
        self.resid4 = ResIdentity(64, 64, 5)
        self.resid5 = ResIdentity(64, 64, 5)

        self.up1 = Upsample(64, 32, 7)
        self.up2 = Upsample(32, 16, 9)
        self.up3 = Upsample(16, 8, 11)
        self.up4 = Upsample(8, 4, 13)
        self.up5 = UpsampleNoactivation(4, 3, 15)

    def _encoder(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = torch.transpose(self.pad1(torch.transpose(x, 1, 2)), 1, 2)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.resid1(x)
        x = self.resid2(x)
        x = self.resid3(x)
        x = self.resid4(x)
        x = self.resid5(x)
        return x

    def _decoder(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = x[:, 1:-1, :]
        x = self.up4(x)
        x = self.up5(x)
        return x

    def forward(self, x):
        f = self._encoder(x)
        y = self._decoder(f)
        return f, y


class RepresentationLearningSingleAutoencoder(nn.Module):
    name = "representation_learning_autoencoder"

    def __init__(self, input_noise_std=1e-6):
        super().__init__()
        self.input_noise_std = input_noise_std
        self.normalize1 = NormalizeStd()
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd()
        self.autoencoder = AutoencoderBlock()
        self.bn = nn.BatchNorm1d(64, eps=BN_EPS, momentum=BN_MOMENTUM, affine=False)

    def _bn_latent(self, f):
        f = torch.transpose(f, 1, 2)
        f = self.bn(f)
        return torch.transpose(f, 1, 2)

    def forward(self, x):
        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        f, y = self.autoencoder(x)
        f = self._bn_latent(f)

        loss = torch.mean(l2_distance(x, y), dim=0)
        return f, y, loss


class RepresentationLearningDenoisingSingleAutoencoder(nn.Module):
    name = "representation_learning_denoising_autoencoder"

    def __init__(self, input_noise_std=1e-6, denoising_noise_std=2e-1):
        super().__init__()
        self.input_noise_std = input_noise_std
        self.denoising_noise_std = denoising_noise_std
        self.normalize1 = NormalizeStd()
        self.add_noise_input = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd()
        self.add_noise_denoising = AddNoise(stddev=self.denoising_noise_std)
        self.normalize_denoising = NormalizeStd()
        self.autoencoder = AutoencoderBlock()
        self.bn = nn.BatchNorm1d(64, eps=BN_EPS, momentum=BN_MOMENTUM, affine=False)

    def _bn_latent(self, f):
        f = torch.transpose(f, 1, 2)
        f = self.bn(f)
        return torch.transpose(f, 1, 2)

    def forward(self, x):
        x = self.normalize1(x)
        x = self.add_noise_input(x)
        x = self.normalize2(x)

        if self.training:
            x_noised = self.normalize_denoising(self.add_noise_denoising(x))
        else:
            x_noised = x

        f, y = self.autoencoder(x_noised)
        f = self._bn_latent(f)

        loss = torch.mean(l2_distance(x, y), dim=0)
        return f, y, loss


class RepresentationLearningMultipleAutoencoder(nn.Module):
    name = "representation_learning_autoencoder_ensemble"

    def __init__(self, input_noise_std=1e-6, eps=1e-27):
        super().__init__()
        self.input_noise_std = input_noise_std
        self.eps = eps
        self.normalize1 = NormalizeStd()
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd()

        self.autoencoders = nn.ModuleList([AutoencoderBlock() for _ in range(5)])

        self.linears = nn.ParameterList(
            [nn.Parameter(_glorot_normal((64, 64))) for _ in range(5)]
        )

        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(64, eps=BN_EPS, momentum=BN_MOMENTUM, affine=False) for _ in range(5)]
        )

    def _bn_latent(self, bn, f):
        f = torch.transpose(f, 1, 2)
        f = bn(f)
        return torch.transpose(f, 1, 2)

    def forward(self, x):
        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        fs = []
        ys = []
        for ae in self.autoencoders:
            f, y = ae(x)
            fs.append(f)
            ys.append(y)

        fps = []
        for i in range(5):
            fp = torch.transpose(
                torch.matmul(self.linears[i], torch.transpose(fs[i].detach(), 1, 2)),
                1,
                2,
            )
            fp = self._bn_latent(self.bns[i], fp)
            fps.append(fp)

        reconstruction_loss = self._get_reconstruction_loss(x, ys)
        ensemble_distance_loss = self._get_ensemble_distance_loss(fps)
        loss = reconstruction_loss + ensemble_distance_loss

        return fps[0], fps[1], fps[2], fps[3], fps[4], ys[0], ys[1], ys[2], ys[3], ys[4], loss

    def _get_ensemble_distance_loss(self, fps):
        total = 0.0
        for i in range(5):
            for j in range(i + 1, 5):
                total = total + torch.mean(self._l2_after_channel_normalizing(fps[i], fps[j]), dim=0)
        return total / 10.0

    def _get_reconstruction_loss(self, x, ys):
        total = 0.0
        for y in ys:
            total = total + torch.mean(l2_distance(x, y), dim=0)
        return total / 5.0

    def _l2_after_channel_normalizing(self, x, y):
        x = demean(x, axis=1)
        y = demean(y, axis=1)
        x = l2_normalize(x, axis=1)
        y = l2_normalize(y, axis=1)
        return torch.sqrt(torch.mean(torch.square(x - y), dim=(1, 2)))


class RepresentationLearningMultipleAutoencoderL4(RepresentationLearningMultipleAutoencoder):
    name = "representation_learning_autoencoderl4_ensemble"

    def _get_ensemble_distance_loss(self, fps):
        total = 0.0
        for i in range(5):
            for j in range(i + 1, 5):
                total = total + torch.mean(self._l4_after_channel_normalizing(fps[i], fps[j]), dim=0)
        return total / 10.0

    def _l4_after_channel_normalizing(self, x, y):
        x = demean(x, axis=1)
        y = demean(y, axis=1)
        x = l4_normalize(x, axis=1)
        y = l4_normalize(y, axis=1)
        return torch.sqrt(torch.sqrt(torch.mean(torch.square(torch.square(x - y)), dim=(1, 2))))


def _glorot_normal(shape):
    t = torch.empty(*shape)
    nn.init.xavier_normal_(t)
    return t
