import h5py
import numpy as np
import pandas as pd
from os import replace
from os.path import exists
from scipy.signal import butter, detrend, sosfiltfilt

from seisbench_data_generator import SeisBenchBatchGenerator, SeisBenchDataGenerator
from config import BATCH_SIZE, SAMPLING_FREQ, FREQMIN, FREQMAX


class ButterworthBatchGenerator(SeisBenchBatchGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sos = butter(
            4, (self.freqmin, self.freqmax),
            btype="bandpass", fs=self.sampling_freq, output="sos",
        )

    def _get_batchx(self, batch_waveforms):
        x = []
        crop_offsets = []

        n_ts_dataset = self._get_ts(self.dataset_time_window)

        for waveform in batch_waveforms:
            wf = self._get_source_waveform(waveform)
            wf = wf[:n_ts_dataset, :]
            if wf.shape[0] < n_ts_dataset:
                raise ValueError(
                    f"trace too short after resampling: sb_index {waveform['sb_index']} "
                    f"({waveform['dataset_source']} dataset), got {wf.shape[0]} samples, "
                    f"expected {n_ts_dataset}"
                )
            x.append(wf)
            crop_offsets.append(int(waveform["crop_offset"]))

        x = np.array(x, dtype=np.float64)
        x = detrend(x, axis=1, type="linear")
        x = sosfiltfilt(self.sos, x, axis=1).astype(np.float32)

        n_ts_model = self._get_ts(self.model_time_window)
        cropped = np.empty((len(x), n_ts_model, x.shape[2]), dtype=np.float32)
        for i, offset in enumerate(crop_offsets):
            cropped[i] = x[i, offset:offset + n_ts_model, :]

        cropped = self._normalize(cropped, axis=1)

        return cropped


class ButterworthDataGenerator(SeisBenchDataGenerator):

    def _render_dataset(self):
        tmp_path = self.processed_hdf5_path + ".tmp"
        with h5py.File(tmp_path, "w") as processed_hdf5:
            for chunk_idx in range(len(self.chunk_metadata_list)):
                bg = ButterworthBatchGenerator(
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
