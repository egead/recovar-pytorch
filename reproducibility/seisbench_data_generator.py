import h5py
import numpy as np
import pandas as pd
from os import replace
from os.path import exists
from torch.utils.data import Dataset

from config import BATCH_SIZE, SAMPLING_FREQ, FREQMIN, FREQMAX


class SeisBenchBatchGenerator:
    def __init__(
        self,
        event_dataset,
        noise_dataset,
        batch_size=BATCH_SIZE,
        batch_metadata=pd.DataFrame(),
        dataset_time_window=120.0,
        model_time_window=30.0,
        sampling_freq=SAMPLING_FREQ,
        freqmin=FREQMIN,
        freqmax=FREQMAX,
        last_axis="channels",
    ):
        self.event_dataset = event_dataset
        self.noise_dataset = noise_dataset
        self.batch_size = batch_size
        self.dataset_time_window = dataset_time_window
        self.model_time_window = model_time_window
        self.sampling_freq = sampling_freq
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.last_axis = last_axis

        self.f = np.fft.fftfreq(
            self._get_ts(self.dataset_time_window), 1.0 / sampling_freq
        )

        self.waveforms = self._get_waveforms(batch_metadata)

    def num_batches(self):
        return len(self.waveforms) // self.batch_size

    def get_batch(self, idx):
        batch_waveforms = self.waveforms[
            (idx * self.batch_size) : ((idx + 1) * self.batch_size)
        ]

        x_batch = self._get_batchx(batch_waveforms)
        y_batch = self._get_batchy(batch_waveforms)

        return x_batch, y_batch

    def _get_waveforms(self, df):
        waveforms = []

        for __, row in df.iterrows():
            waveform = {
                "sb_index": int(row["sb_index"]),
                "crop_offset": row["crop_offset"],
                "label": row["label"],
                "dataset_source": row.get("dataset_source", row["label"]),
            }
            waveforms.append(waveform)

        return waveforms

    def _get_source_waveform(self, waveform):
        if waveform["dataset_source"] == "eq":
            wf = self.event_dataset.get_waveforms(
                waveform["sb_index"], sampling_rate=self.sampling_freq
            )
        else:
            wf = self.noise_dataset.get_waveforms(
                waveform["sb_index"], sampling_rate=self.sampling_freq
            )

        wf = wf[:3, :]
        return np.transpose(wf, axes=[1, 0])

    def _get_batchx(self, batch_waveforms):
        x = []
        crop_offsets = []

        n_ts_dataset = self._get_ts(self.dataset_time_window)

        for waveform in batch_waveforms:
            wf = self._get_source_waveform(waveform)

            wf = wf[:n_ts_dataset, :]

            x.append(wf)
            crop_offsets.append(waveform["crop_offset"])

        x = np.array(x).astype(np.float32)
        crop_offsets = np.array(crop_offsets).astype(np.float32)

        crop_offsets = np.expand_dims(np.expand_dims(crop_offsets, axis=1), axis=2)

        f = np.expand_dims(np.expand_dims(self.f, axis=0), axis=2)

        if self.last_axis == "timesteps":
            x = np.transpose(x, axes=[0, 2, 1])

        xw = np.fft.fft(x, axis=1)

        xw = xw * np.exp(1j * 2 * np.pi * f * crop_offsets / self.sampling_freq)

        mask = (np.abs(self.f) < self.freqmin) | (np.abs(self.f) > self.freqmax)
        xw[:, mask, :] = 0

        x = np.fft.ifft(xw, axis=1)
        x = np.real(x).astype(np.float32)

        x = x[:, 0 : self._get_ts(self.model_time_window), :]

        x = x - np.mean(x, axis=1, keepdims=True)
        x = self._normalize(x, axis=1)

        return x

    def _get_batchy(self, batch_waveforms):
        y = np.array(
            [waveform["label"] == "eq" for waveform in batch_waveforms],
            dtype=np.int32,
        )
        return y

    def _get_ts(self, t):
        return int(t * self.sampling_freq)

    @staticmethod
    def _normalize(x, axis):
        norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
        return x / (1e-37 + norm)


class SeisBenchDataGenerator(Dataset):
    def __init__(
        self,
        processed_hdf5_path,
        chunk_metadata_list,
        batch_size,
        event_dataset,
        noise_dataset,
        phase_ensured_crop_ratio,
        dataset_time_window=120.0,
        model_time_window=30.0,
        sampling_freq=SAMPLING_FREQ,
        active_chunks=[],
        freqmin=FREQMIN,
        freqmax=FREQMAX,
        last_axis="channels",
        *args,
        **kwargs
    ):
        self.processed_hdf5_path = processed_hdf5_path
        self.chunk_metadata_list = chunk_metadata_list
        self.batch_size = batch_size
        self.event_dataset = event_dataset
        self.noise_dataset = noise_dataset
        self.phase_ensured_crop_ratio = phase_ensured_crop_ratio
        self.dataset_time_window = dataset_time_window
        self.model_time_window = model_time_window
        self.sampling_freq = sampling_freq
        self.active_chunks = active_chunks
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.last_axis = last_axis
        self.chunk_batch_counts = self.get_chunk_batch_counts()

        if not exists(self.processed_hdf5_path):
            self._render_dataset()

        self.processed_hdf5 = h5py.File(self.processed_hdf5_path, "r", locking=True)

    def getitem(self, idx):
        chunk_idx, batch_offset = self.get_chunk_idx_and_batch_offset(idx)

        x_batch = self.processed_hdf5.get(
            "data/x/chunk{}/{}".format(chunk_idx, batch_offset)
        )[...]
        y_batch = self.processed_hdf5.get(
            "data/y/chunk{}/{}".format(chunk_idx, batch_offset)
        )[...]

        return x_batch, y_batch

    def length(self):
        n_batches = 0
        for chunk in self.active_chunks:
            n_batches = n_batches + self.chunk_batch_counts[chunk]

        return n_batches

    def on_epoch_end(self):
        pass

    def get_chunk_idx_and_batch_offset(self, batch_idx):
        batch_offset = batch_idx
        for chunk in self.active_chunks:
            if batch_offset < self.chunk_batch_counts[chunk]:
                return chunk, batch_offset

            batch_offset -= self.chunk_batch_counts[chunk]

        return None, None

    def get_chunk_batch_counts(self):
        chunk_batch_counts = {}
        for chunk_idx, chunk_metadata in enumerate(self.chunk_metadata_list):
            chunk_batch_counts[chunk_idx] = len(chunk_metadata) // self.batch_size

        return chunk_batch_counts

    def __del__(self):
        if hasattr(self, "processed_hdf5"):
            self.processed_hdf5.close()

    def _render_dataset(self):
        tmp_path = self.processed_hdf5_path + ".tmp"
        with h5py.File(tmp_path, "w") as processed_hdf5:
            for chunk_idx in range(len(self.chunk_metadata_list)):
                bg = SeisBenchBatchGenerator(
                    event_dataset=self.event_dataset,
                    noise_dataset=self.noise_dataset,
                    batch_size=self.batch_size,
                    batch_metadata=self.chunk_metadata_list[chunk_idx],
                    dataset_time_window=self.dataset_time_window,
                    model_time_window=self.model_time_window,
                    sampling_freq=self.sampling_freq,
                    freqmin=self.freqmin,
                    freqmax=self.freqmax,
                    last_axis=self.last_axis,
                )

                n_chunk_batches = bg.num_batches()

                for chunk_batch_offset in range(n_chunk_batches):
                    x, y = bg.get_batch(chunk_batch_offset)

                    processed_hdf5.create_dataset(
                        "data/x/chunk{}/{}".format(chunk_idx, chunk_batch_offset),
                        data=x,
                        compression=None,
                    )

                    processed_hdf5.create_dataset(
                        "data/y/chunk{}/{}".format(chunk_idx, chunk_batch_offset),
                        data=y,
                        compression=None,
                    )

                processed_hdf5.create_dataset(
                    "metadata/chunk{}/{}".format(chunk_idx, "num_batches"),
                    data=n_chunk_batches,
                )

        replace(tmp_path, self.processed_hdf5_path)
