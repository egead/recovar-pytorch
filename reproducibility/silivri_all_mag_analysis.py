import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/mnt/second_drive/seisbench")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import seisbench.models as sbm
import torch
from obspy import UTCDateTime


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from recovar_torch import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder


ROOT = Path("/mnt/data_a/ege")
SILIVRI_OUTPUT = ROOT / "RECOVAR_SILIVRI2019" / "output"
METADATA = SILIVRI_OUTPUT / "SILIVRI2019_metadata.csv"
WAVEFORMS = SILIVRI_OUTPUT / "SILIVRI2019_waveforms.hdf5"
RAW = Path("/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/prepared_waveforms/day_by_day")
PICKS = Path("/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv")
CATALOG_CANDIDATES = [
    Path("/mnt/second_drive/ege/recovar/silivri_durand_catalog.txt"),
    Path.home() / "recovar/silivri_durand_catalog.txt",
]
CHECKPOINT_CANDIDATES = [
    Path("/mnt/second_drive/ege/picovar/models/recovar_instance_seisbench_benchmark.pt"),
    Path.home() / "picovar/models/recovar_instance_seisbench_benchmark.pt",
]
PARTIAL_CACHE = REPO / "silivri_all_instance_scores_partial.npz"
ANALYSIS_CACHE = REPO / "silivri_all_instance_mag_cache.npz"
OUTPUT = REPO / "silivri_all_magnitude_analysis"
SAMPLING_RATE = 100
SOURCE_SAMPLES = 6000
WINDOW_SAMPLES = 3000
BATCH_SIZE = 256
NOISE_SAMPLE_SIZE = 40000
MAGNITUDE_BIN_WIDTH = 0.5
HISTOGRAM_STEP = 0.25
BOOTSTRAP_REPETITIONS = 1000
FALSE_POSITIVE_RATES = [0.001, 0.01]
MATCH_TOLERANCE_SECONDS = 0.05


def existing_path(candidates, description):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{description} not found in: {[str(path) for path in candidates]}")


def parse_mixed_datetime(values):
    try:
        return pd.to_datetime(values, format="mixed")
    except (TypeError, ValueError):
        return values.map(lambda value: pd.to_datetime(value) if pd.notna(value) else pd.NaT)


def prepare_picks():
    picks = pd.read_csv(PICKS)
    catalog = pd.read_csv(
        existing_path(CATALOG_CANDIDATES, "Durand catalog"),
        sep=r"\s+",
        skiprows=1,
        header=None,
        usecols=[1, 2, 3, 4, 5, 6, 11],
        names=["year", "month", "day", "hour", "minute", "second", "magnitude"],
    )
    catalog["catalog_orgtime"] = pd.to_datetime(catalog[["year", "month", "day", "hour", "minute"]]) + pd.to_timedelta(catalog["second"], unit="s")
    picks["pick_orgtime"] = parse_mixed_datetime(picks["orgtime"])
    picks["event_id"] = picks["pick_orgtime"].astype(str)
    picks["p_arrival_time"] = parse_mixed_datetime(picks["p_arrival_time"])
    picks = pd.merge_asof(
        picks.sort_values("pick_orgtime"),
        catalog[["catalog_orgtime", "magnitude"]].sort_values("catalog_orgtime"),
        left_on="pick_orgtime",
        right_on="catalog_orgtime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    )
    return picks.dropna(subset=["station", "p_arrival_time", "event_id", "magnitude"])


def prepare_descriptors():
    metadata = pd.read_csv(METADATA).drop(columns=["Unnamed: 0", "index"], errors="ignore")
    metadata["metadata_index"] = np.arange(len(metadata))
    metadata["trace_start"] = parse_mixed_datetime(metadata["trace_start_time"])
    metadata["p_time"] = metadata["trace_start"] + pd.to_timedelta(pd.to_numeric(metadata["p_arrival_sample"], errors="coerce") / SAMPLING_RATE, unit="s")
    events = metadata.loc[metadata["label"].eq("eq") & metadata["p_time"].notna()].copy()
    events["station_key"] = events["station_name"].astype(str)
    picks = prepare_picks().copy()
    picks["station_key"] = picks["station"].astype(str)
    events = pd.merge_asof(
        events.sort_values("p_time"),
        picks[["station_key", "p_arrival_time", "event_id", "magnitude"]].sort_values("p_arrival_time"),
        left_on="p_time",
        right_on="p_arrival_time",
        by="station_key",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_SECONDS),
    )
    events = events.dropna(subset=["event_id", "magnitude"]).copy()
    event_pre_seconds = 6.0 + ((events["metadata_index"].to_numpy(dtype=np.int64) * 7) % 181) / 10.0
    events["crop_offset"] = np.clip(
        np.rint(pd.to_numeric(events["p_arrival_sample"]).to_numpy() - event_pre_seconds * SAMPLING_RATE),
        0,
        SOURCE_SAMPLES - WINDOW_SAMPLES,
    ).astype(int)
    events["kind"] = "event"
    noise = metadata.loc[metadata["label"].eq("no")].copy()
    sample_size = min(NOISE_SAMPLE_SIZE, len(noise))
    noise = noise.sample(n=sample_size, random_state=0).copy()
    noise["crop_offset"] = (
        noise["metadata_index"].to_numpy(dtype=np.int64) * 2654435761
    ) % (SOURCE_SAMPLES - WINDOW_SAMPLES + 1)
    noise["event_id"] = ""
    noise["magnitude"] = np.nan
    noise["kind"] = "noise"
    noise["noise_role"] = np.where(np.arange(len(noise)) % 2 == 0, "calibration", "validation")
    events["noise_role"] = ""
    keep = [
        "metadata_index",
        "trace_name",
        "station_name",
        "trace_start",
        "crop_offset",
        "event_id",
        "magnitude",
        "kind",
        "noise_role",
    ]
    descriptors = pd.concat([events[keep], noise[keep]], ignore_index=True)
    descriptors["window_start"] = descriptors["trace_start"] + pd.to_timedelta(descriptors["crop_offset"] / SAMPLING_RATE, unit="s")
    descriptors = descriptors.sort_values(["station_name", "window_start", "trace_name"]).reset_index(drop=True)
    return descriptors, picks


def station_directories():
    directories = {}
    for directory in RAW.iterdir():
        if directory.is_dir():
            directories[directory.name.upper()] = directory
    return directories


def waveform_index(directory, station):
    entries = []
    paths = sorted(set(directory.rglob("*.mseed")) | set(directory.rglob("*.miniseed")))
    if not paths:
        raise FileNotFoundError(f"no MiniSEED files found in {directory}")
    for path in paths:
        try:
            stream = obspy.read(str(path), headonly=True)
        except Exception:
            continue
        matching = [trace for trace in stream if trace.stats.station.upper() == station]
        if matching:
            entries.append((path, min(trace.stats.starttime for trace in matching), max(trace.stats.endtime for trace in matching)))
    if not entries:
        raise RuntimeError(f"no readable MiniSEED headers found for station {station}")
    return entries


def raw_window(entries, station, start, stream_cache):
    start = UTCDateTime(start.to_pydatetime())
    end = start + WINDOW_SAMPLES / SAMPLING_RATE
    paths = [path for path, file_start, file_end in entries if file_end >= start and file_start <= end]
    if not paths:
        raise RuntimeError(f"no raw waveform overlaps {station} at {start}")
    stream = obspy.Stream()
    for path in paths:
        if path not in stream_cache:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                stream_cache[path] = obspy.read(str(path))
            for warning in caught:
                print(f"MiniSEED warning in {path}: {warning.message}")
            stream_cache.move_to_end(path)
            while len(stream_cache) > 6:
                stream_cache.popitem(last=False)
        stream += stream_cache[path].copy().trim(start - 1, end + 1)
    stream = obspy.Stream([trace for trace in stream if trace.stats.station.upper() == station])
    stream.merge(method=1, fill_value=None)
    components = []
    for component in "ZNE":
        candidates = [trace.copy() for trace in stream if trace.stats.channel.upper().endswith(component)]
        if not candidates:
            raise RuntimeError(f"missing {component} component for {station} at {start}")
        trace = max(candidates, key=lambda candidate: candidate.stats.npts)
        trace.trim(start - 1, end + 1)
        if abs(trace.stats.sampling_rate - SAMPLING_RATE) > 1e-6:
            trace.resample(SAMPLING_RATE)
        trace.trim(start, end - 1 / SAMPLING_RATE, pad=True, fill_value=None, nearest_sample=False)
        if np.ma.isMaskedArray(trace.data) and np.ma.getmaskarray(trace.data).any():
            raise RuntimeError(f"data gap in {component} component for {station} at {start}")
        data = np.asarray(trace.data, dtype=np.float32)
        if len(data) != WINDOW_SAMPLES:
            raise RuntimeError(f"incomplete {component} component for {station} at {start}")
        components.append(data[:WINDOW_SAMPLES])
    window = np.stack(components)
    if not np.isfinite(window).all():
        raise RuntimeError(f"non-finite raw waveform for {station} at {start}")
    return window


def recovar_window(hdf5, trace_name, crop_offset):
    data = np.asarray(hdf5[f"data/{trace_name}"][...], dtype=np.float32)
    if data.shape[0] != SOURCE_SAMPLES and data.shape[1] == SOURCE_SAMPLES:
        data = data.T
    data = data[:, [2, 1, 0]]
    data = data[crop_offset:crop_offset + WINDOW_SAMPLES]
    data -= data.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(np.square(data), axis=0, keepdims=True))
    return data / (1e-37 + norm)


def phasenet_preprocess(windows):
    output = windows.astype(np.float32)
    output -= output.mean(axis=2, keepdims=True)
    output /= 1e-10 + np.max(np.abs(output), axis=2, keepdims=True)
    return np.pad(output, ((0, 0), (0, 0), (0, 1)))


def load_models(device):
    representation = RepresentationLearningMultipleAutoencoder().to(device)
    state = torch.load(existing_path(CHECKPOINT_CANDIDATES, "RECOVAR checkpoint"), map_location=device, weights_only=False)
    representation.load_state_dict(state)
    recovar = ClassifierMultipleAutoencoder(representation).to(device).eval()
    phasenet = sbm.PhaseNet.from_pretrained("instance").to(device).eval()
    noise_index = list(phasenet.labels).index("N")
    return recovar, phasenet, noise_index


def descriptor_signature(descriptors):
    return np.asarray(
        [f"{row.trace_name}|{row.crop_offset}" for row in descriptors.itertuples()],
        dtype=str,
    )


def load_partial(descriptors):
    count = len(descriptors)
    recovar_scores = np.full(count, np.nan, dtype=np.float32)
    phasenet_scores = np.full(count, np.nan, dtype=np.float32)
    states = np.zeros(count, dtype=np.int8)
    exclusion_reasons = np.full(count, "", dtype="<U512")
    signature = descriptor_signature(descriptors)
    if PARTIAL_CACHE.exists():
        partial = np.load(PARTIAL_CACHE)
        if len(partial["recovar_scores"]) != count or not np.array_equal(partial["signature"].astype(str), signature):
            raise ValueError(f"partial cache does not match the current Silivri descriptors: {PARTIAL_CACHE}")
        recovar_scores[:] = partial["recovar_scores"]
        phasenet_scores[:] = partial["phasenet_scores"]
        if "states" in partial:
            states[:] = partial["states"]
        else:
            states[np.isfinite(recovar_scores) & np.isfinite(phasenet_scores)] = 1
        if "exclusion_reasons" in partial:
            exclusion_reasons[:] = partial["exclusion_reasons"].astype(str)
    return recovar_scores, phasenet_scores, states, exclusion_reasons, signature


def save_partial(recovar_scores, phasenet_scores, states, exclusion_reasons, signature):
    temporary = PARTIAL_CACHE.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        recovar_scores=recovar_scores,
        phasenet_scores=phasenet_scores,
        states=states,
        exclusion_reasons=exclusion_reasons,
        signature=signature,
    )
    temporary.replace(PARTIAL_CACHE)


def score_descriptors(descriptors):
    recovar_scores, phasenet_scores, states, exclusion_reasons, signature = load_partial(descriptors)
    if np.all(states != 0):
        print(f"Detection scores loaded from {PARTIAL_CACHE}")
        return recovar_scores, phasenet_scores, states, exclusion_reasons
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    recovar, phasenet, noise_index = load_models(device)
    directories = station_directories()
    indices = {}
    stream_cache = OrderedDict()
    total_batches = int(np.ceil(len(descriptors) / BATCH_SIZE))
    with h5py.File(WAVEFORMS, "r") as hdf5, torch.inference_mode():
        for batch_index, begin in enumerate(range(0, len(descriptors), BATCH_SIZE)):
            end = min(begin + BATCH_SIZE, len(descriptors))
            if np.all(states[begin:end] != 0):
                print(f"Detection completed:{batch_index + 1}/{total_batches} cached")
                continue
            recovar_windows = []
            raw_windows = []
            valid_indices = []
            for descriptor_index, row in zip(range(begin, end), descriptors.iloc[begin:end].itertuples()):
                if states[descriptor_index] != 0:
                    continue
                station = str(row.station_name).upper()
                if station not in directories:
                    candidates = [path for key, path in directories.items() if station in key]
                    if len(candidates) != 1:
                        raise FileNotFoundError(f"could not identify raw waveform directory for station {station}")
                    directory = candidates[0]
                else:
                    directory = directories[station]
                if station not in indices:
                    indices[station] = waveform_index(directory, station)
                try:
                    raw = raw_window(indices[station], station, row.window_start, stream_cache)
                    recovar_data = recovar_window(hdf5, row.trace_name, int(row.crop_offset))
                except (KeyError, RuntimeError, ValueError) as error:
                    states[descriptor_index] = -1
                    exclusion_reasons[descriptor_index] = str(error)
                    print(f"Excluded {row.trace_name}: {error}")
                    continue
                recovar_windows.append(recovar_data)
                raw_windows.append(raw)
                valid_indices.append(descriptor_index)
            if valid_indices:
                recovar_input = torch.from_numpy(np.stack(recovar_windows)).float().to(device)
                recovar_scores[valid_indices] = recovar(recovar_input).cpu().numpy()
                phasenet_input = torch.from_numpy(phasenet_preprocess(np.stack(raw_windows))).to(device)
                phasenet_output = phasenet(phasenet_input)
                if isinstance(phasenet_output, (tuple, list)):
                    phasenet_output = phasenet_output[0]
                phasenet_scores[valid_indices] = (1.0 - phasenet_output[:, noise_index, :]).amax(dim=1).cpu().numpy()
                states[valid_indices] = 1
            save_partial(recovar_scores, phasenet_scores, states, exclusion_reasons, signature)
            print(f"Detection completed:{batch_index + 1}/{total_batches} excluded:{int((states == -1).sum())}")
    return recovar_scores, phasenet_scores, states, exclusion_reasons


def build_analysis_cache():
    descriptors, picks = prepare_descriptors()
    print(f"matched event station records: {(descriptors['kind'] == 'event').sum()}")
    print(f"noise calibration/validation windows: {(descriptors['kind'] == 'noise').sum()}")
    recovar_scores, phasenet_scores, states, exclusion_reasons = score_descriptors(descriptors)
    valid = states == 1
    events = descriptors.loc[descriptors["kind"].eq("event") & valid].copy()
    event_mask = descriptors["kind"].eq("event").to_numpy() & valid
    noise_calibration = descriptors["noise_role"].eq("calibration").to_numpy() & valid
    noise_validation = descriptors["noise_role"].eq("validation").to_numpy() & valid
    print(f"excluded raw windows: {(states == -1).sum()}")
    excluded = descriptors.loc[states == -1, ["trace_name", "station_name", "window_start", "kind"]].copy()
    excluded["reason"] = exclusion_reasons[states == -1]
    excluded.to_csv(OUTPUT / "silivri_excluded_windows.csv", index=False)
    start = descriptors["trace_start"].min()
    end = descriptors["trace_start"].max()
    truth = picks.loc[picks["pick_orgtime"].between(start, end)].groupby("event_id")["magnitude"].first()
    np.savez_compressed(
        ANALYSIS_CACHE,
        event_ids=events["event_id"].astype(str).to_numpy(dtype=str),
        event_magnitudes=events["magnitude"].to_numpy(dtype=float),
        recovar_event_scores=recovar_scores[event_mask],
        phasenet_event_scores=phasenet_scores[event_mask],
        recovar_noise_calibration=recovar_scores[noise_calibration],
        phasenet_noise_calibration=phasenet_scores[noise_calibration],
        recovar_noise_validation=recovar_scores[noise_validation],
        phasenet_noise_validation=phasenet_scores[noise_validation],
        catalog_magnitudes=truth.to_numpy(dtype=float),
        matched_event_magnitudes=events.groupby("event_id")["magnitude"].first().to_numpy(dtype=float),
    )
    return np.load(ANALYSIS_CACHE)


def magnitude_edges(magnitudes):
    lower = np.floor(np.nanmin(magnitudes) / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH
    upper = np.ceil(np.nanmax(magnitudes) / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH
    edges = np.arange(lower, upper + MAGNITUDE_BIN_WIDTH * 1.01, MAGNITUDE_BIN_WIDTH)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def clustered_metric(frame, columns, thresholds, seed):
    detected = frame.copy()
    for column, threshold in zip(columns, thresholds):
        detected[column] = detected[column].ge(threshold)
    grouped = detected.groupby("event_id")[columns].agg("sum")
    totals = detected.groupby("event_id").size().rename("count")
    grouped = grouped.join(totals)
    points = [grouped[column].sum() / grouped["count"].sum() for column in columns]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((BOOTSTRAP_REPETITIONS, len(columns)), dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    values = [grouped[column].to_numpy(dtype=float) for column in columns]
    for index in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(grouped), len(grouped))
        denominator = counts[selected].sum()
        bootstrap[index] = [value[selected].sum() / denominator for value in values]
    intervals = [np.quantile(bootstrap[:, index], [0.025, 0.975]) for index in range(len(columns))]
    difference = points[0] - points[1]
    difference_interval = np.quantile(bootstrap[:, 0] - bootstrap[:, 1], [0.025, 0.975])
    return points, intervals, difference, difference_interval


def histogram(values, title, filename):
    lower = np.floor(np.nanmin(values) / HISTOGRAM_STEP) * HISTOGRAM_STEP
    upper = np.ceil(np.nanmax(values) / HISTOGRAM_STEP) * HISTOGRAM_STEP
    bins = np.arange(lower, upper + HISTOGRAM_STEP * 1.01, HISTOGRAM_STEP)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(values, bins=bins, color="0.45", edgecolor="white", linewidth=0.7)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of earthquakes")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = np.load(ANALYSIS_CACHE) if ANALYSIS_CACHE.exists() else build_analysis_cache()
    events = pd.DataFrame(
        {
            "event_id": cache["event_ids"].astype(str),
            "magnitude": cache["event_magnitudes"],
            "RECOVAR-INSTANCE": cache["recovar_event_scores"],
            "PhaseNet-INSTANCE": cache["phasenet_event_scores"],
        }
    )
    noise = {
        "RECOVAR-INSTANCE": (cache["recovar_noise_calibration"], cache["recovar_noise_validation"]),
        "PhaseNet-INSTANCE": (cache["phasenet_noise_calibration"], cache["phasenet_noise_validation"]),
    }
    thresholds = {model: {} for model in noise}
    achieved = {model: {} for model in noise}
    for model, (calibration, validation) in noise.items():
        for false_positive_rate in FALSE_POSITIVE_RATES:
            threshold = np.quantile(calibration, 1.0 - false_positive_rate, method="higher")
            thresholds[model][false_positive_rate] = threshold
            achieved[model][false_positive_rate] = np.mean(validation >= threshold)
    edges = magnitude_edges(events["magnitude"].to_numpy())
    rows = []
    differences = []
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        frame = events.loc[events["magnitude"].ge(left) & events["magnitude"].lt(right)]
        if frame.empty:
            continue
        for fpr_index, false_positive_rate in enumerate(FALSE_POSITIVE_RATES):
            columns = ["RECOVAR-INSTANCE", "PhaseNet-INSTANCE"]
            points, intervals, difference, difference_interval = clustered_metric(
                frame,
                columns,
                [thresholds[column][false_positive_rate] for column in columns],
                1000 * bin_index + fpr_index,
            )
            for model_index, model in enumerate(columns):
                rows.append(
                    {
                        "model": model,
                        "false_positive_rate_target": false_positive_rate,
                        "false_positive_rate_achieved": achieved[model][false_positive_rate],
                        "threshold": thresholds[model][false_positive_rate],
                        "magnitude_left": left,
                        "magnitude_right": right,
                        "n_station_records": len(frame),
                        "n_catalog_events": frame["event_id"].nunique(),
                        "recall": points[model_index],
                        "recall_ci_lower": intervals[model_index][0],
                        "recall_ci_upper": intervals[model_index][1],
                    }
                )
            differences.append(
                {
                    "false_positive_rate_target": false_positive_rate,
                    "magnitude_left": left,
                    "magnitude_right": right,
                    "n_station_records": len(frame),
                    "n_catalog_events": frame["event_id"].nunique(),
                    "recall_difference": difference,
                    "difference_ci_lower": difference_interval[0],
                    "difference_ci_upper": difference_interval[1],
                }
            )
    summary = pd.DataFrame(rows)
    difference_summary = pd.DataFrame(differences)
    summary.to_csv(OUTPUT / "silivri_station_recall_by_magnitude.csv", index=False)
    difference_summary.to_csv(OUTPUT / "silivri_station_recall_difference_by_magnitude.csv", index=False)
    bins = summary[["magnitude_left", "magnitude_right"]].drop_duplicates().reset_index(drop=True)
    labels = [f"[{row.magnitude_left:.1f}, {row.magnitude_right:.1f})" for row in bins.itertuples()]
    x = np.arange(len(labels))
    colors = {"RECOVAR-INSTANCE": "#3b6ea8", "PhaseNet-INSTANCE": "#8b5ea7"}
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.8), sharex=True)
    for axis, false_positive_rate in zip(axes, FALSE_POSITIVE_RATES):
        count_rows = summary.loc[summary["false_positive_rate_target"].eq(false_positive_rate) & summary["model"].eq("RECOVAR-INSTANCE")]
        count_axis = axis.twinx()
        count_axis.bar(x, count_rows["n_station_records"], width=0.82, color="0.85", edgecolor="none")
        count_axis.set_ylabel("Available records", color="0.45")
        count_axis.tick_params(axis="y", colors="0.45")
        count_axis.spines["top"].set_visible(False)
        count_axis.set_zorder(0)
        axis.set_zorder(1)
        axis.patch.set_visible(False)
        for model in noise:
            model_rows = summary.loc[summary["false_positive_rate_target"].eq(false_positive_rate) & summary["model"].eq(model)]
            values = model_rows["recall"].to_numpy()
            error = np.vstack([values - model_rows["recall_ci_lower"].to_numpy(), model_rows["recall_ci_upper"].to_numpy() - values])
            axis.errorbar(x, values, yerr=error, marker="o", linewidth=1.7, capsize=2.5, color=colors[model], label=model, zorder=3)
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel("Station-level recall")
        axis.set_title(f"Global Silivri noise FPR = {100 * false_positive_rate:g}%")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Magnitude interval")
    fig.tight_layout()
    fig.savefig(OUTPUT / "silivri_station_recall_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "silivri_station_recall_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    for axis, false_positive_rate in zip(axes, FALSE_POSITIVE_RATES):
        frame = difference_summary.loc[difference_summary["false_positive_rate_target"].eq(false_positive_rate)]
        values = frame["recall_difference"].to_numpy()
        error = np.vstack([values - frame["difference_ci_lower"].to_numpy(), frame["difference_ci_upper"].to_numpy() - values])
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        axis.errorbar(x, values, yerr=error, marker="o", linewidth=1.7, capsize=2.5, color="#2f6b4f")
        axis.set_ylabel("Recall difference")
        axis.set_title(f"RECOVAR − PhaseNet at Silivri noise FPR = {100 * false_positive_rate:g}%")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Magnitude interval")
    fig.tight_layout()
    fig.savefig(OUTPUT / "silivri_station_recall_difference_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "silivri_station_recall_difference_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    histogram(cache["catalog_magnitudes"], "Silivri catalog magnitude distribution", "silivri_catalog_magnitude_histogram")
    histogram(cache["matched_event_magnitudes"], "Matched Silivri event magnitude distribution", "silivri_matched_magnitude_histogram")
    print(OUTPUT / "silivri_station_recall_by_magnitude.pdf")


if __name__ == "__main__":
    main()
