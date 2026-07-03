import json
from os import environ, makedirs

with open("settings.json", "r") as file:
    settings = json.load(file)

if "SEISBENCH_CACHE_ROOT" in settings:
    environ.setdefault("SEISBENCH_CACHE_ROOT", settings["SEISBENCH_CACHE_ROOT"])

import seisbench.data as sbd
from seisbench.data import WaveformDataset

from kfold_trainer import KfoldTrainer
from seisbench_kfold_environment import SeisBenchKFoldEnvironment
from directory import get_checkpoint_dir
from config import INSTANCE_TIME_WINDOW


def get_dataset_time_window(dataset):
    return settings["SEISBENCH_DATASETS"][dataset].get(
        "time_window", INSTANCE_TIME_WINDOW
    )


def load_seisbench_datasets(dataset):
    dataset_config = settings["SEISBENCH_DATASETS"][dataset]
    component_order = dataset_config.get("component_order", "ZNE")

    if "seisbench_name" in dataset_config:
        full_dataset = getattr(sbd, dataset_config["seisbench_name"])(
            sampling_rate=None,
            component_order=component_order,
        )
        noise_mask = full_dataset.metadata["trace_category"] == "noise"
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
            dataset_time_window=get_dataset_time_window(self.dataset),
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
