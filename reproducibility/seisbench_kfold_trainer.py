import json
from os import environ, makedirs

with open("settings.json", "r") as file:
    settings = json.load(file)

if "SEISBENCH_CACHE_ROOT" in settings:
    environ.setdefault("SEISBENCH_CACHE_ROOT", settings["SEISBENCH_CACHE_ROOT"])

import numpy as np
import pandas as pd
import seisbench.data as sbd
from seisbench.data import WaveformDataset

from kfold_trainer import KfoldTrainer
from seisbench_kfold_environment import SeisBenchKFoldEnvironment
from directory import get_checkpoint_dir
from config import INSTANCE_TIME_WINDOW, SAMPLING_FREQ, WINDOW_SIZE


def get_dataset_time_window(dataset, event_dataset=None):
    time_window = settings["SEISBENCH_DATASETS"][dataset].get("time_window")
    if time_window is not None:
        return time_window
    if event_dataset is None:
        return INSTANCE_TIME_WINDOW
    metadata = event_dataset.metadata
    if "trace_npts" not in metadata.columns:
        return INSTANCE_TIME_WINDOW
    fs = metadata.get("trace_sampling_rate_hz", pd.Series(SAMPLING_FREQ, index=metadata.index))
    seconds = metadata["trace_npts"] / fs
    return float(np.clip(round(np.nanmedian(seconds)), WINDOW_SIZE, 180.0))


def load_seisbench_datasets(dataset):
    dataset_config = settings["SEISBENCH_DATASETS"][dataset]
    component_order = dataset_config.get("component_order", "ZNE")

    if "seisbench_name" in dataset_config:
        full_dataset = getattr(sbd, dataset_config["seisbench_name"])(
            sampling_rate=None,
            component_order=component_order,
        )
        filter_split = dataset_config.get("filter_split")
        if filter_split is not None and "split" in full_dataset.metadata.columns:
            full_dataset.filter(
                full_dataset.metadata["split"].isin(filter_split), inplace=True
            )
        if "trace_category" in full_dataset.metadata.columns:
            noise_mask = full_dataset.metadata["trace_category"] == "noise"
        else:
            noise_mask = pd.Series(False, index=full_dataset.metadata.index)
        event_dataset = full_dataset.filter(~noise_mask, inplace=False)
        noise_dataset = full_dataset.filter(noise_mask, inplace=False)
        return event_dataset, noise_dataset

    event_dataset = WaveformDataset(
        dataset_config["event_path"],
        sampling_rate=None,
        component_order=component_order,
    )
    noise_dataset = WaveformDataset(
        dataset_config["noise_path"],
        sampling_rate=None,
        component_order=component_order,
    )
    return event_dataset, noise_dataset


class SeisBenchKfoldTrainer(KfoldTrainer):
    def __init__(self, *args, event_dataset=None, noise_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        if event_dataset is None or noise_dataset is None:
            event_dataset, noise_dataset = load_seisbench_datasets(self.dataset)
        self.event_dataset = event_dataset
        self.noise_dataset = noise_dataset

    def train(self):
        kfold_env = SeisBenchKFoldEnvironment(
            dataset=self.dataset,
            event_dataset=self.event_dataset,
            noise_dataset=self.noise_dataset,
            dataset_time_window=get_dataset_time_window(self.dataset, self.event_dataset),
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed,
        )

        (
            train_gen,
            validation_gen,
            __,
            __,
        ) = kfold_env.get_generators(self.split)

        makedirs(
            get_checkpoint_dir(
                self.exp_name, self.model_name, self.dataset, self.split
            ),
            exist_ok=True,
        )

        model = self._create_model()

        fit_result = self._train_model(
            model=model,
            split=self.split,
            train_gen=train_gen,
            validation_gen=validation_gen,
        )

        self._save_history(self.split, fit_result)
