import os
import obspy
from continuous_data_processor import ContinuousDataPreprocessor

catalog = "/mnt/second_drive/recovar/reproducibility/catalog_balikesir_recovar_2025-07-01_2026-02-28.csv"
waveforms_dir = "/mnt/second_drive/recovar-balikesir/data/continuous/"

station_dirs = []
for (root, dirs, files) in os.walk(waveforms_dir):
    for dir in dirs:
        station_dirs.append(os.path.join(waveforms_dir, dir))


preprocessor = ContinuousDataPreprocessor(
    catalog_csv=catalog,
    output_hdf5_path=f"output/BALIKESIR_continuous_waveforms.hdf5",
    output_metadata_csv_path=f"output/BALIKESIR_continuous_metadata.csv",
    window_length=60,
    sampling_rate=100,
)
for station in station_dirs:
    preprocessor.process_station(station)
