from config import (
    BATCH_SIZE,
    KFOLD_SPLITS,
    DATASET_CHUNKS,
    INSTANCE_TIME_WINDOW,
    STEAD_TIME_WINDOW,
    SAMPLING_FREQ,
    FREQMIN,
    FREQMAX,
    TRAIN_VALIDATION_SPLIT,
    TEST_RATIO,
    SUBSAMPLING_FACTOR,
    PHASE_PICK_ENSURED_CROP_RATIO,
    PHASE_ENSURING_MARGIN,
    APPLY_RESAMPLING,
    RESAMPLE_EQ_RATIO
)
from directory import (
    STEAD_WAVEFORMS_HDF5_PATH,
    STEAD_METADATA_CSV_PATH,
    INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
    INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
    INSTANCE_EQ_METADATA_CSV_PATH,
    INSTANCE_NOISE_METADATA_CSV_PATH,
    PREPROCESSED_DATASET_DIRECTORY,
)
import json

from data_generator import (
    DataGenerator,
    GeneratorWrapper,
    PredictGenerator,
)
from os.path import join, exists
from os import makedirs
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold

class KFoldEnvironment:
    def __init__(
        self,
        dataset,
        preprocessed_dataset_directory=PREPROCESSED_DATASET_DIRECTORY,
        batch_size=BATCH_SIZE,
        stead_time_window=STEAD_TIME_WINDOW,
        instance_time_window=INSTANCE_TIME_WINDOW,
        stead_waveforms_hdf5=STEAD_WAVEFORMS_HDF5_PATH,
        stead_metadata_csv=STEAD_METADATA_CSV_PATH,
        instance_eq_waveforms_hdf5=INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
        instance_no_waveforms_hdf5=INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
        instance_eq_metadata_csv=INSTANCE_EQ_METADATA_CSV_PATH,
        instance_no_metadata_csv=INSTANCE_NOISE_METADATA_CSV_PATH,
        model_time_window=30.0,
        phase_ensured_crop_ratio=PHASE_PICK_ENSURED_CROP_RATIO,
        phase_ensuring_margin=PHASE_ENSURING_MARGIN,
        n_splits=KFOLD_SPLITS,
        n_chunks=DATASET_CHUNKS,
        subsampling_factor=SUBSAMPLING_FACTOR,
        sampling_freq=SAMPLING_FREQ,
        train_val_ratio=TRAIN_VALIDATION_SPLIT,
        test_ratio= TEST_RATIO,
        apply_resampling=APPLY_RESAMPLING,
        resample_while_keeping_total_waveforms_fixed=False,
        resample_eq_ratio=RESAMPLE_EQ_RATIO,
        freqmin=FREQMIN,
        freqmax=FREQMAX,
    ):
        with open("settings.json", 'r') as file:
            settings = json.load(file)

        self.preprocessed_dataset_directory = preprocessed_dataset_directory
        self.model_time_window = model_time_window
        self.phase_ensured_crop_ratio = phase_ensured_crop_ratio
        self.phase_ensuring_margin = phase_ensuring_margin
        self.stead_time_window = stead_time_window
        self.instance_time_window = instance_time_window
        self.subsampling_factor = subsampling_factor
        self.sampling_freq = sampling_freq
        self.train_val_ratio = train_val_ratio
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.resample_while_keeping_total_waveforms_fixed = resample_while_keeping_total_waveforms_fixed
        self.apply_resampling = apply_resampling
        self.resample_eq_ratio = resample_eq_ratio
        self.n_chunks = n_chunks
        self.test_ratio = test_ratio
        self._batch_size = batch_size
        self._n_splits = n_splits
        self._dataset = dataset

        if dataset == "stead":
            metadata = self._parse_stead_metadata(stead_metadata_csv)

            self.eq_hdf5_path = stead_waveforms_hdf5
            self.no_hdf5_path = stead_waveforms_hdf5
            self.last_axis = "channels"
            self.dataset_time_window = self.stead_time_window

        if dataset in settings.get("CUSTOM_DATASETS", {}):
            dataset_config = settings["CUSTOM_DATASETS"][dataset]
            metadata = self._parse_stead_metadata(dataset_config["metadata"])

            self.eq_hdf5_path = dataset_config["waveforms"]
            self.no_hdf5_path = dataset_config["waveforms"]
            self.last_axis = "channels"
            self.dataset_time_window = self.stead_time_window

        elif dataset == "instance":
            metadata = self._parse_instance_metadata(
                instance_eq_metadata_csv, instance_no_metadata_csv
            )

            self.eq_hdf5_path = instance_eq_waveforms_hdf5
            self.no_hdf5_path = instance_no_waveforms_hdf5
            self.last_axis = "timesteps"
            self.dataset_time_window = self.instance_time_window

        train_and_val_split_chunks, test_split_chunks = self._form_kfold_splits()

        (
            self.train_splits,
            self.validation_splits,
        ) = self._seperate_train_and_validation_chunks(train_and_val_split_chunks)

        self.test_splits = test_split_chunks

        chunk_metadata_list = self._split_dataset_to_chunks(metadata, "source_id")

        if self.apply_resampling:
            chunk_metadata_list = self._resample(chunk_metadata_list)

        chunk_metadata_list = self._subsample_chunk_metadata(chunk_metadata_list)

        chunk_metadata_list = self._make_chunk_metadata_multiple_of_batch_size(
            chunk_metadata_list
        )

        chunk_metadata_list = [
            self._assign_crop_offsets(chunk_metadata)
            for chunk_metadata in chunk_metadata_list
        ]

        chunk_metadata_list = [
            self._assign_chunk_idx(chunk_metadata, chunk_idx)
            for chunk_idx, chunk_metadata in enumerate(chunk_metadata_list)
        ]

        makedirs(join(self.preprocessed_dataset_directory, self.dataset), exist_ok=True)
        identifier = self.dataset
        if self.apply_resampling:
            metadata_path = join(
                self.preprocessed_dataset_directory,
                self.dataset,
                f"{identifier}_resampled_eq{int(100 * self.resample_eq_ratio)}_subsampled_{int(100 * self.subsampling_factor)}percent.csv"
            )
        else:
            metadata_path = join(
                self.preprocessed_dataset_directory,
                self.dataset,
                f"{identifier}_subsampled_{int(100 * self.subsampling_factor)}percent.csv"
            )

        if not exists(metadata_path):
            pd.concat(chunk_metadata_list).to_csv(metadata_path)

        self.chunk_metadata_list = chunk_metadata_list

    @property
    def dataset(self):
        return self._dataset

    @property
    def n_splits(self):
        return self._n_splits

    @property
    def batch_size(self):
        return self._batch_size

    def get_generators(self, split):
        train_datagen = self._get_datagen(self.train_splits[split])
        validation_datagen = self._get_datagen(self.validation_splits[split])
        test_datagen = self._get_datagen(self.test_splits[split])
        predict_datagen = self._get_datagen(self.test_splits[split])

        train_gen = GeneratorWrapper(train_datagen)
        validation_gen = GeneratorWrapper(validation_datagen)
        test_gen = GeneratorWrapper(test_datagen)
        predict_gen = PredictGenerator(predict_datagen)

        return train_gen, validation_gen, test_gen, predict_gen

    def get_split_metadata(self, split):
        train_metadata = self.get_chunklist_metadata(self.train_splits[split])
        validation_metadata = self.get_chunklist_metadata(self.validation_splits[split])
        test_metadata = self.get_chunklist_metadata(self.test_splits[split])

        return train_metadata, validation_metadata, test_metadata

    def get_batch_metadata(self, split, operation, batch_idx):
        if operation == "train":
            split_chunks = self.train_splits[split]
        elif operation == "validation":
            split_chunks = self.validation_splits[split]
        elif operation == "test":
            split_chunks = self.test_splits[split]

        datagen = self._get_datagen(split_chunks)

        chunk_idx, batch_offset = datagen.get_chunk_idx_and_batch_offset(batch_idx)
        return self.chunk_metadata_list[chunk_idx].iloc[
            batch_offset * self.batch_size : (batch_offset + 1) * self.batch_size
        ]

    def get_chunklist_metadata(self, chunks):
        metadata_list = []
        for chunk in chunks:
            metadata_list.append(self.chunk_metadata_list[chunk])

        if len(metadata_list) > 0:
            return pd.concat(metadata_list)
        else:
            return pd.DataFrame()

    def _split_dataset_to_chunks(self, dataset_metadata, colname, random_state=0):
        unique_vals = list(set(list(dataset_metadata[colname].values)))
        unique_vals.sort()

        random.seed(random_state)
        sampling_size = len(unique_vals) // self.n_chunks

        chunks = []
        for i in range(self.n_chunks):
            chunks.append(
                random.sample(unique_vals, min(sampling_size, len(unique_vals)))
            )
            unique_vals = list(set(unique_vals) - set(chunks[-1]))
            unique_vals.sort()

        chunk_metadata_list = []
        for chunk in chunks:
            chunk_metadata_list.append(
                dataset_metadata[dataset_metadata[colname].isin(chunk)]
            )

        return chunk_metadata_list

    def _form_kfold_splits(self, random_state=0):
        chunks = range(self.n_chunks)

        if self._n_splits > 1:
            kfold = KFold(shuffle=True, random_state=random_state, n_splits=self.n_splits)
            kfold_gen = kfold.split(chunks)

            training_chunk_idxs_for_each_split = []
            testing_chunk_idxs_for_each_split = []

            for train_chunk_idxs, test_chunks_idx in kfold_gen:
                training_chunk_idxs_for_each_split.append(train_chunk_idxs)
                testing_chunk_idxs_for_each_split.append(test_chunks_idx)
        else:
            chunk_list = list(chunks)
            random.seed(random_state)
            random.shuffle(chunk_list)

            num_training_chunks = int(len(chunk_list) * (1. - self.test_ratio))
            training_chunks = chunk_list[0:num_training_chunks]
            testing_chunks = chunk_list[num_training_chunks:]

            training_chunk_idxs_for_each_split = [training_chunks]
            testing_chunk_idxs_for_each_split = [testing_chunks]

        return training_chunk_idxs_for_each_split, testing_chunk_idxs_for_each_split

    def _parse_stead_metadata(self, metadata_csv):
        metadata = pd.read_csv(metadata_csv)

        metadata["source_id"] = metadata["source_id"].astype(str)
        metadata.rename({"receiver_code": "station_name"}, axis=1, inplace=True)

        eq_metadata = metadata[metadata.trace_category == "earthquake_local"].copy()
        no_metadata = metadata[metadata.trace_category == "noise"].copy()

        no_metadata["source_id"] = no_metadata["trace_name"]

        eq_metadata.loc[:, "label"] = "eq"
        no_metadata.loc[:, "label"] = "no"

        standardized_metadata = pd.concat([eq_metadata, no_metadata])

        return standardized_metadata

    def _parse_instance_metadata(self, eq_metadata_csv, no_metadata_csv):
        eq_metadata = pd.read_csv(eq_metadata_csv)
        no_metadata = pd.read_csv(no_metadata_csv)

        eq_metadata["source_id"] = eq_metadata["source_id"].astype(str)
        eq_metadata.rename(
            columns={
                "station_code": "station_name",
                "trace_P_arrival_sample": "p_arrival_sample",
                "trace_S_arrival_sample": "s_arrival_sample",
            },
            inplace=True,
        )
        no_metadata.rename(columns={"station_code": "station_name"}, inplace=True)

        no_metadata["source_id"] = no_metadata["trace_name"]

        eq_metadata["label"] = "eq"
        no_metadata["label"] = "no"

        standardized_metadata = pd.concat([eq_metadata, no_metadata])
        return standardized_metadata

    def _make_chunk_metadata_multiple_of_batch_size(self, chunk_metadata_list):
        cropped_chunk_metadata_list = []
        for chunk_metadata in chunk_metadata_list:
            cropped_chunk_metadata_list.append(
                chunk_metadata[
                    0 : ((len(chunk_metadata) // self.batch_size) * self.batch_size)
                ]
            )

        return cropped_chunk_metadata_list

    def _get_datagen(self, active_chunks=None):
        processed_hdf5_dir = join(
            PREPROCESSED_DATASET_DIRECTORY,
            self.dataset,
        )
        makedirs(processed_hdf5_dir, exist_ok=True)

        identifier = self.dataset
        if self.apply_resampling:
            processed_hdf5_path = join(
                processed_hdf5_dir,
                f"{identifier}_resampled_eq{int(100 * self.resample_eq_ratio)}_subsampled_{int(100 * self.subsampling_factor)}percent.hdf5"
            )
        else:
            processed_hdf5_path = join(
                processed_hdf5_dir,
                f"{identifier}_subsampled_{int(100 * self.subsampling_factor)}percent.hdf5"
            )

        datagen = DataGenerator(
            processed_hdf5_path=processed_hdf5_path,
            chunk_metadata_list=self.chunk_metadata_list,
            batch_size=self.batch_size,
            eq_hdf5_path=self.eq_hdf5_path,
            no_hdf5_path=self.no_hdf5_path,
            dataset_time_window=self.dataset_time_window,
            model_time_window=self.model_time_window,
            phase_ensured_crop_ratio=self.phase_ensured_crop_ratio,
            last_axis=self.last_axis,
            sampling_freq=self.sampling_freq,
            active_chunks=active_chunks,
            freqmin=self.freqmin,
            freqmax=self.freqmax,
        )

        return datagen

    def _seperate_train_and_validation_chunks(self, chunk_metadata_list):
        train_chunk_dfs = []
        validation_chunk_dfs = []
        random.seed(0)

        for i in range(len(chunk_metadata_list)):
            n_train_chunks = int(len(chunk_metadata_list[i]) * self.train_val_ratio)

            random.shuffle(chunk_metadata_list[i])
            train_chunk_dfs.append(chunk_metadata_list[i][0:n_train_chunks])
            validation_chunk_dfs.append(chunk_metadata_list[i][n_train_chunks:])

        return train_chunk_dfs, validation_chunk_dfs

    def _subsample_chunk_metadata(self, chunk_metadata_list):
        subsampled_chunk_metadata_list = []

        for chunk_metadata in chunk_metadata_list:
            subsampled_chunk_metadata_list.append(
                chunk_metadata.sample(frac=self.subsampling_factor, random_state=0)
            )

        return subsampled_chunk_metadata_list

    def _resample(self, chunk_metadata_list):
        resampled_chunk_metadata_list = []

        for chunk_metadata in chunk_metadata_list:
            eq_chunk_metadata = chunk_metadata[chunk_metadata.label == "eq"]
            no_chunk_metadata = chunk_metadata[chunk_metadata.label == "no"]

            num_eqs = len(eq_chunk_metadata)
            num_nos = len(no_chunk_metadata)
            total_samples = num_eqs + num_nos

            target_num_eqs = int(total_samples * self.resample_eq_ratio)
            target_num_nos = total_samples - target_num_eqs

            eq_resampled = eq_chunk_metadata.sample(n=target_num_eqs, replace=True, random_state=0)
            no_resampled = no_chunk_metadata.sample(n=target_num_nos, replace=True, random_state=0)

            resampled_metadata = pd.concat([eq_resampled, no_resampled], axis=0)
            resampled_chunk_metadata_list.append(resampled_metadata)

        return resampled_chunk_metadata_list

    def _assign_chunk_idx(self, metadata, chunk_idx):
        metadata["chunk_idx"] = chunk_idx

        return metadata

    def _assign_crop_offsets(self, metadata):
        metadata.reset_index(drop=True, inplace=True)

        metadata["crop_offset_low_limit"] = 0
        metadata["crop_offset_high_limit"] = self._get_ts(
            self.dataset_time_window
        ) - self._get_ts(self.model_time_window)

        metadata["crop_offset_min"] = metadata["crop_offset_low_limit"]
        metadata["crop_offset_max"] = metadata["crop_offset_high_limit"]

        metadata_eq = metadata[metadata.label == "eq"]
        metadata_no = metadata[metadata.label == "no"]

        n_pick_ensured_eq_waveforms = int(
            self.phase_ensured_crop_ratio * len(metadata_eq)
        )

        metadata_eq_pick_ensured_crop = metadata_eq[0:n_pick_ensured_eq_waveforms]
        metadata_eq_random_crop = metadata_eq[n_pick_ensured_eq_waveforms:]

        metadata_eq_pick_ensured_crop = self._assign_pick_ensured_crop_offset_ranges(
            metadata_eq_pick_ensured_crop
        )

        metadata = pd.concat(
            [metadata_eq_pick_ensured_crop, metadata_eq_random_crop, metadata_no],
            axis=0,
        )

        np.random.seed(0)
        metadata["crop_offset"] = metadata["crop_offset_min"] + (
            metadata["crop_offset_max"] - metadata["crop_offset_min"]
        ) * np.random.uniform(0, 1, len(metadata))

        metadata["crop_offset"] = metadata["crop_offset"].astype(int)

        metadata.drop(
            [
                "crop_offset_low_limit",
                "crop_offset_high_limit",
                "crop_offset_min",
                "crop_offset_max",
            ],
            axis=1,
            inplace=True,
        )

        metadata.sort_index(inplace=True)

        return metadata

    def _assign_pick_ensured_crop_offset_ranges(self, eq_metadata):
        _eq_metadata = eq_metadata.copy()

        _eq_metadata["crop_offset_min"] = (
            _eq_metadata[["p_arrival_sample", "s_arrival_sample"]].min(axis=1)
            + self._get_ts(self.phase_ensuring_margin)
            - self._get_ts(self.model_time_window)
        )

        _eq_metadata["crop_offset_min"] = _eq_metadata[
            ["crop_offset_min", "crop_offset_low_limit"]
        ].max(axis=1)

        _eq_metadata["crop_offset_max"] = _eq_metadata[
            ["p_arrival_sample", "s_arrival_sample"]
        ].max(axis=1) - self._get_ts(self.phase_ensuring_margin)

        _eq_metadata["crop_offset_max"] = _eq_metadata[
            ["crop_offset_max", "crop_offset_high_limit"]
        ].min(axis=1)

        return _eq_metadata

    def _get_ts(self, t):
        return int(t * self.sampling_freq)
