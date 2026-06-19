import torch
import torch.nn as nn
import numpy as np
from recovar_torch.config import BATCH_SIZE, SAMPLING_FREQ
from recovar_torch.layers import CrossCovarianceCircular
from recovar_torch.cubic_interpolation import cubic_interp1d
from recovar_torch.utils import l2_normalize
from itertools import combinations

N_TIMESTEPS = 3000
N_CHANNELS = 3


def gaussian_window(timesteps, sigma=1.25, axis=1):
    t = np.expand_dims(np.linspace(-timesteps / 2.0, timesteps / 2.0, timesteps), axis=0)
    g = np.exp(-np.power(t, 2.0) / (2 * np.power(sigma, 2.0)))
    return g / np.sum(g, axis=axis, keepdims=True)


def eq_metric(fcov):
    n_timesteps = fcov.shape[1]
    g = torch.as_tensor(gaussian_window(n_timesteps), dtype=fcov.dtype, device=fcov.device)
    z = torch.clamp(torch.mean(fcov * g, dim=1), min=0)
    return 1.0 - torch.exp(-z)


class ClassifierAutocovariance(nn.Module):
    name = "autocovariance"

    def __init__(self, model=None, method_params=None):
        super().__init__()
        self.model = model
        self.cc = CrossCovarianceCircular()

    def forward(self, inputs):
        out = self.model(inputs)
        f = out[0]
        fcov = self.cc(f, f)
        return eq_metric(fcov)


class ClassifierAugmentedAutoencoder(nn.Module):
    name = "classifier_augmentation_cross_covariances"
    TIME_WINDOW = 30

    def __init__(self, model=None, method_params=None):
        super().__init__()
        if method_params is None:
            method_params = {"augmentations": 5, "std": 0.15, "knots": 4}
        self.model = model
        self.method_params = method_params
        self.cc = CrossCovarianceCircular()

        self.register_buffer(
            "t",
            torch.linspace(0.0, self.TIME_WINDOW * ((N_TIMESTEPS - 1.0) / N_TIMESTEPS), N_TIMESTEPS),
        )
        self.register_buffer(
            "t_knots",
            torch.linspace(
                0.0,
                self.TIME_WINDOW * ((N_TIMESTEPS - 1.0) / N_TIMESTEPS),
                self.method_params["knots"] + 2,
            ),
        )

    def _timewarp(self, x):
        knot_values = torch.randn(self.method_params["knots"]) * self.method_params["std"]
        knot_values = torch.cat([torch.zeros(1), knot_values, torch.zeros(1)], dim=0)

        t_warped = self.t + cubic_interp1d(self.t, self.t_knots, knot_values)
        t_warped = t_warped - torch.min(t_warped, keepdim=True, dim=0).values
        t_warped = (
            (self.TIME_WINDOW - (2.0 / SAMPLING_FREQ)) / torch.max(t_warped, keepdim=True, dim=0).values
        ) * t_warped

        idx_floor = torch.floor(t_warped * SAMPLING_FREQ)
        interp_point_distance_to_floor = (t_warped * SAMPLING_FREQ) - idx_floor

        weight_floor = 1.0 / (1e-37 + interp_point_distance_to_floor)
        weight_ceil = 1.0 / (1e-37 + (1.0 - interp_point_distance_to_floor))

        weight_floor = weight_floor / (weight_floor + weight_ceil)
        weight_floor = torch.unsqueeze(torch.unsqueeze(weight_floor, 0), 2)

        idx_floor = idx_floor.to(torch.int64)
        x = weight_floor * torch.index_select(x, 1, idx_floor) + (
            1.0 - weight_floor
        ) * torch.index_select(x, 1, idx_floor + 1)
        return x

    def _cross_covariance_ensemble_mean(self, f_list):
        covariances = []
        for pair in combinations(list(range(len(f_list))), 2):
            covariances.append(self.cc(f_list[pair[0]], f_list[pair[1]]))
        return torch.mean(torch.stack(covariances), dim=0)

    def forward(self, inputs):
        x = l2_normalize(inputs)

        augmented_f = [None] * self.method_params["augmentations"]
        for i in range(self.method_params["augmentations"]):
            augmented_x = self._timewarp(x)
            out = self.model(augmented_x)
            augmented_f[i] = out[0]

        fcov = self._cross_covariance_ensemble_mean(augmented_f)
        return eq_metric(fcov)


class ClassifierMultipleAutoencoder(nn.Module):
    name = "representation_cross_covariances"

    def __init__(self, model=None, method_params=None):
        super().__init__()
        self.model = model
        self.cc = CrossCovarianceCircular()

    def _cross_covariance_ensemble_mean(self, f_list):
        covariances = []
        for pair in combinations(list(range(len(f_list))), 2):
            covariances.append(self.cc(f_list[pair[0]], f_list[pair[1]]))
        return torch.mean(torch.stack(covariances), dim=0)

    def forward(self, inputs):
        out = self.model(inputs)
        f1, f2, f3, f4, f5 = out[0], out[1], out[2], out[3], out[4]
        fcov = self._cross_covariance_ensemble_mean([f1, f2, f3, f4, f5])
        return eq_metric(fcov)
