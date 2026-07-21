import os
import shutil
import sys
from os import makedirs
from os.path import join
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from recovar_torch import RepresentationLearningMultipleAutoencoder
from seisbench_kfold_trainer import (
    SeisBenchKfoldTrainer,
    get_dataset_time_window,
)
from seisbench_kfold_environment import SeisBenchKFoldEnvironment
from seisbench_data_generator_butterworth import ButterworthDataGenerator
from directory import get_checkpoint_path, get_history_csv_path, get_checkpoint_dir


BUTTERWORTH_PREPROCESSED_DIR = "/mnt/data_a/ege/recovar_data_preprocessed_butterworth"


def find_picovar_repo():
    candidates = [
        Path.home() / "picovar",
        Path("/mnt/second_drive/ege/picovar"),
    ]
    if "PICOVAR_DIR" in os.environ:
        candidates.insert(0, Path(os.environ["PICOVAR_DIR"]))
    for candidate in candidates:
        if (candidate / "picovar").is_dir():
            return candidate
    raise FileNotFoundError("picovar repo not found; set PICOVAR_DIR")


PICOVAR_MODELS_DIR = find_picovar_repo() / "models"


class ButterworthKFoldEnvironment(SeisBenchKFoldEnvironment):

    def _get_datagen(self, active_chunks=None):
        processed_hdf5_dir = join(
            self.preprocessed_dataset_directory,
            self.dataset,
        )
        makedirs(processed_hdf5_dir, exist_ok=True)

        identifier = self.dataset
        if self.apply_resampling:
            processed_hdf5_path = join(
                processed_hdf5_dir,
                "{}_resampled_eq{}_subsampled_{}percent.hdf5".format(
                    identifier,
                    int(100 * self.resample_eq_ratio),
                    int(100 * self.subsampling_factor),
                ),
            )
        else:
            processed_hdf5_path = join(
                processed_hdf5_dir,
                "{}_subsampled_{}percent.hdf5".format(
                    identifier,
                    int(100 * self.subsampling_factor),
                ),
            )

        datagen = ButterworthDataGenerator(
            processed_hdf5_path=processed_hdf5_path,
            chunk_metadata_list=self.chunk_metadata_list,
            batch_size=self.batch_size,
            event_dataset=self.event_dataset,
            noise_dataset=self.noise_dataset,
            dataset_time_window=self.dataset_time_window,
            model_time_window=self.model_time_window,
            phase_ensured_crop_ratio=self.phase_ensured_crop_ratio,
            sampling_freq=self.sampling_freq,
            active_chunks=active_chunks,
            freqmin=self.freqmin,
            freqmax=self.freqmax,
            last_axis=self.last_axis,
        )

        return datagen


class ButterworthKfoldTrainer(SeisBenchKfoldTrainer):

    def train(self):
        kfold_env = ButterworthKFoldEnvironment(
            dataset=self.dataset,
            event_dataset=self.event_dataset,
            noise_dataset=self.noise_dataset,
            preprocessed_dataset_directory=BUTTERWORTH_PREPROCESSED_DIR,
            dataset_time_window=get_dataset_time_window(
                self.dataset, self.event_dataset
            ),
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=(
                self.resample_while_keeping_total_waveforms_fixed
            ),
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


if __name__ == "__main__":
    DATASET = "instance"
    NUM_EPOCHS = 20
    SPLIT = 0

    EXP_NAME = "recovar_{}_butterworth".format(DATASET)
    OUT_PATH = PICOVAR_MODELS_DIR / "recovar_{}_butterworth_seisbench_benchmark.pt".format(DATASET)

    if OUT_PATH.exists():
        print("{} already exists, skipping".format(OUT_PATH))
    else:
        trainer = ButterworthKfoldTrainer(
            EXP_NAME,
            RepresentationLearningMultipleAutoencoder,
            DATASET,
            SPLIT,
            epochs=NUM_EPOCHS,
            apply_resampling=False,
        )
        trainer.train()

        history = pd.read_csv(
            get_history_csv_path(EXP_NAME, trainer.model_name, DATASET, SPLIT)
        )
        if history["val_loss"].isna().all():
            print("Training produced no valid loss, NOT copying a model")
        else:
            best_epoch = int(history["val_loss"].idxmin())
            checkpoint = get_checkpoint_path(
                EXP_NAME, trainer.model_name, DATASET, SPLIT, best_epoch
            )
            PICOVAR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(checkpoint, OUT_PATH)
            print(
                "best epoch {} val_loss {:.6f} -> {}".format(
                    best_epoch, history["val_loss"].min(), OUT_PATH
                )
            )
