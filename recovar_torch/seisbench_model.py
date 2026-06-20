from __future__ import annotations

from typing import Any

import torch

import seisbench.models as sbm
import seisbench.util as sbu

from recovar_torch.representation_learning_models import (
    RepresentationLearningMultipleAutoencoder,
    N_TIMESTEPS,
)
from recovar_torch.classifier_models import ClassifierMultipleAutoencoder
from recovar_torch.config import SAMPLING_FREQ


class RecovarDetector(sbm.WaveformModel):
    """RECOVAR ensemble autoencoder as a SeisBench sliding-window detector.

    The classifier emits one earthquake score in [0, 1] per 30 s, 3-component
    window. This matches SeisBench ``output_type="point"`` model. The window step is
    the ``stride`` annotate argument, in samples at 100 Hz (100 = 1 s).
    """

    _weight_warnings = []

    _annotate_args = {
        **sbm.WaveformModel._annotate_args,
        "earthquake_threshold": (
            "Score threshold in [0, 1] above which a window is reported as a detection "
            "(used by classify).",
            0.5,
        ),
    }

    def __init__(
        self,
        stride: int = 100,
        detection_threshold: float = 0.5,
        input_noise_std: float = 1e-6,
        eps: float = 1e-27,
        **kwargs,
    ):
        super().__init__(
            output_type="point",
            in_samples=N_TIMESTEPS,
            pred_sample=N_TIMESTEPS // 2,
            labels=["earthquake"],
            sampling_rate=SAMPLING_FREQ,
            component_order="ZNE",
            filter_args=("bandpass",),
            filter_kwargs={"freqmin": 1.0, "freqmax": 20.0, "corners": 4, "zerophase": True},
            default_args={
                "stride": stride,
                "earthquake_threshold": detection_threshold,
            },
            **kwargs,
        )

        self.representation = RepresentationLearningMultipleAutoencoder(
            input_noise_std=input_noise_std,
            eps=eps,
        )
        self.classifier = ClassifierMultipleAutoencoder(model=self.representation)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch.transpose(1, 2)
        batch = batch - batch.mean(dim=1, keepdim=True)
        return batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.classifier(x)
        return score.unsqueeze(-1)

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        threshold = argdict.get(
            "earthquake_threshold", self._annotate_args.get("earthquake_threshold")[1]
        )
        detections = self.picks_from_annotations(
            annotations.select(channel=f"{self.__class__.__name__}_earthquake"),
            threshold,
            "earthquake",
        )
        return sbu.ClassifyOutput(self.name, detections=sbu.PickList(sorted(detections)))

    def load_representation_state_dict(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.representation.load_state_dict(state)
        return self
