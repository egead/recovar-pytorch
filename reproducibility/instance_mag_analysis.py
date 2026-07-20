import os
import sys
from pathlib import Path

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/mnt/second_drive/seisbench")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seisbench.data as sbd
import seisbench.models as sbm
import torch
from scipy.signal import butter, sosfiltfilt


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from recovar_torch import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder


CHECKPOINT_CANDIDATES = [
    Path("/mnt/second_drive/ege/picovar/models/recovar_instance_seisbench_benchmark.pt"),
    Path.home() / "picovar/models/recovar_instance_seisbench_benchmark.pt",
]
CACHE = REPO / "instance_mag_cache.npz"
PARTIAL_CACHE = REPO / "instance_mag_scores_partial.npz"
OUTPUT = REPO / "instance_magnitude_analysis"
SAMPLING_RATE = 100
WINDOW_SAMPLES = 3000
BATCH_SIZE = 256
MAGNITUDE_BIN_WIDTH = 0.5
HISTOGRAM_STEP = 0.25
BOOTSTRAP_REPETITIONS = 1000
FALSE_POSITIVE_RATES = [0.001, 0.01]
PHASE_COLUMNS = [
    "trace_p_arrival_sample",
    "trace_P_arrival_sample",
    "trace_P1_arrival_sample",
    "trace_Pg_arrival_sample",
    "trace_Pn_arrival_sample",
    "trace_s_arrival_sample",
    "trace_S_arrival_sample",
    "trace_S1_arrival_sample",
    "trace_Sg_arrival_sample",
    "trace_Sn_arrival_sample",
]


def checkpoint_path():
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(f"RECOVAR checkpoint not found in: {[str(path) for path in CHECKPOINT_CANDIDATES]}")


def magnitude_column(metadata):
    preferred = ["source_magnitude", "source_magnitude_mw", "source_local_magnitude", "magnitude"]
    for name in preferred:
        if name in metadata.columns and pd.to_numeric(metadata[name], errors="coerce").notna().any():
            return name
    candidates = [name for name in metadata.columns if "magnitude" in name.lower() or name.lower() == "mag"]
    for name in candidates:
        if pd.to_numeric(metadata[name], errors="coerce").notna().any():
            return name
    raise KeyError(f"no numeric magnitude column found; available columns: {metadata.columns.tolist()}")


def first_arrival(metadata):
    values = []
    for name in PHASE_COLUMNS:
        value = metadata.get(name)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
    return min(values) if values else np.nan


def test_descriptors(metadata):
    phase_columns = [name for name in PHASE_COLUMNS if name in metadata.columns]
    if not phase_columns:
        raise KeyError("INSTANCE metadata has no phase-arrival columns")
    test = metadata.loc[metadata["split"].eq("test")].copy()
    arrivals = test[phase_columns].apply(pd.to_numeric, errors="coerce")
    test["first_arrival"] = arrivals.min(axis=1, skipna=True)
    test["label"] = np.where(test["first_arrival"].notna(), "eq", "no")
    test["dataset_index"] = test.index.astype(int)
    test["magnitude"] = pd.to_numeric(test[magnitude_column(metadata)], errors="coerce")
    if "source_id" not in test.columns:
        test["source_id"] = test["trace_name"].astype(str)
    else:
        missing_source = test["source_id"].isna()
        test.loc[missing_source, "source_id"] = test.loc[missing_source, "trace_name"].astype(str)
        test["source_id"] = test["source_id"].astype(str)
    return test.reset_index(drop=True)


def window_start(npts, first_arrival, descriptor_index):
    maximum = max(0, npts - WINDOW_SAMPLES)
    if np.isfinite(first_arrival):
        pre_seconds = 5.0 + ((descriptor_index * 7) % 101) / 10.0
        return int(np.clip(round(first_arrival - pre_seconds * SAMPLING_RATE), 0, maximum))
    if maximum == 0:
        return 0
    return int((descriptor_index * 2654435761) % (maximum + 1))


def recovar_preprocess(windows):
    sos = butter(4, [1.0, 20.0], btype="band", fs=SAMPLING_RATE, output="sos")
    output = sosfiltfilt(sos, windows.astype(np.float64), axis=2)
    output -= output.mean(axis=2, keepdims=True)
    output /= 1e-12 + output.std(axis=2, keepdims=True)
    return np.transpose(output.astype(np.float32), (0, 2, 1))


def phasenet_preprocess(windows):
    output = windows.astype(np.float32)
    output -= output.mean(axis=2, keepdims=True)
    output /= 1e-12 + np.max(np.abs(output), axis=2, keepdims=True)
    return np.pad(output, ((0, 0), (0, 0), (0, 1)))


def load_models(device):
    representation = RepresentationLearningMultipleAutoencoder().to(device)
    state = torch.load(checkpoint_path(), map_location=device, weights_only=False)
    representation.load_state_dict(state)
    recovar = ClassifierMultipleAutoencoder(representation).to(device).eval()
    phasenet = sbm.PhaseNet.from_pretrained("instance").to(device).eval()
    noise_index = list(phasenet.labels).index("N")
    return recovar, phasenet, noise_index


def load_partial(count):
    recovar_scores = np.full(count, np.nan, dtype=np.float32)
    phasenet_scores = np.full(count, np.nan, dtype=np.float32)
    if PARTIAL_CACHE.exists():
        partial = np.load(PARTIAL_CACHE)
        if len(partial["recovar_scores"]) != count:
            raise ValueError(f"partial cache length differs from current official test split: {PARTIAL_CACHE}")
        recovar_scores[:] = partial["recovar_scores"]
        phasenet_scores[:] = partial["phasenet_scores"]
    return recovar_scores, phasenet_scores


def save_partial(recovar_scores, phasenet_scores):
    temporary = PARTIAL_CACHE.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, recovar_scores=recovar_scores, phasenet_scores=phasenet_scores)
    temporary.replace(PARTIAL_CACHE)


def score_test_set(dataset, descriptors):
    recovar_scores, phasenet_scores = load_partial(len(descriptors))
    if np.isfinite(recovar_scores).all() and np.isfinite(phasenet_scores).all():
        print(f"Detection scores loaded from {PARTIAL_CACHE}")
        return recovar_scores, phasenet_scores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    recovar, phasenet, noise_index = load_models(device)
    total_batches = int(np.ceil(len(descriptors) / BATCH_SIZE))
    with torch.inference_mode():
        for batch_index, begin in enumerate(range(0, len(descriptors), BATCH_SIZE)):
            end = min(begin + BATCH_SIZE, len(descriptors))
            if np.isfinite(recovar_scores[begin:end]).all() and np.isfinite(phasenet_scores[begin:end]).all():
                print(f"Detection completed:{batch_index + 1}/{total_batches} cached")
                continue
            windows = []
            for descriptor_index in range(begin, end):
                row = descriptors.iloc[descriptor_index]
                waveform, metadata = dataset.get_sample(int(row["dataset_index"]))
                waveform = np.asarray(waveform[:3], dtype=np.float32)
                if waveform.shape[0] != 3:
                    raise ValueError(f"trace {row['trace_name']} does not have three components")
                start = window_start(waveform.shape[1], first_arrival(metadata), descriptor_index)
                window = waveform[:, start:start + WINDOW_SAMPLES]
                if window.shape[1] < WINDOW_SAMPLES:
                    window = np.pad(window, ((0, 0), (0, WINDOW_SAMPLES - window.shape[1])))
                windows.append(window)
            windows = np.stack(windows)
            recovar_input = torch.from_numpy(recovar_preprocess(windows)).to(device)
            recovar_scores[begin:end] = recovar(recovar_input).cpu().numpy()
            phase_input = torch.from_numpy(phasenet_preprocess(windows)).to(device)
            phase_output = phasenet(phase_input)
            if isinstance(phase_output, (tuple, list)):
                phase_output = phase_output[0]
            phasenet_scores[begin:end] = (1.0 - phase_output[:, noise_index, :]).amax(dim=1).cpu().numpy()
            save_partial(recovar_scores, phasenet_scores)
            print(f"Detection completed:{batch_index + 1}/{total_batches}")
    return recovar_scores, phasenet_scores


def build_cache():
    dataset = sbd.InstanceCountsCombined(sampling_rate=SAMPLING_RATE, component_order="ZNE")
    descriptors = test_descriptors(dataset.metadata)
    print(f"official INSTANCE test windows: {len(descriptors)}")
    recovar_scores, phasenet_scores = score_test_set(dataset, descriptors)
    event_rows = descriptors["label"].eq("eq") & descriptors["magnitude"].notna()
    events = descriptors.loc[event_rows, ["source_id", "magnitude"]].copy()
    events["RECOVAR-INSTANCE"] = recovar_scores[event_rows.to_numpy()]
    events["PhaseNet-INSTANCE"] = phasenet_scores[event_rows.to_numpy()]
    noise_rows = descriptors["label"].eq("no").to_numpy()
    catalog = dataset.metadata.copy()
    catalog["magnitude"] = pd.to_numeric(catalog[magnitude_column(catalog)], errors="coerce")
    if "source_id" in catalog.columns:
        catalog_events = catalog.dropna(subset=["source_id", "magnitude"]).groupby("source_id")["magnitude"].first()
    else:
        catalog_events = catalog["magnitude"].dropna()
    test_event_magnitudes = events.groupby("source_id")["magnitude"].first().to_numpy(dtype=float)
    np.savez_compressed(
        CACHE,
        analysis_level=np.asarray("station_window_v1"),
        event_source_ids=events["source_id"].astype(str).to_numpy(dtype=str),
        event_magnitudes=events["magnitude"].to_numpy(dtype=float),
        recovar_event_scores=events["RECOVAR-INSTANCE"].to_numpy(dtype=float),
        phasenet_event_scores=events["PhaseNet-INSTANCE"].to_numpy(dtype=float),
        recovar_noise_scores=recovar_scores[noise_rows],
        phasenet_noise_scores=phasenet_scores[noise_rows],
        full_magnitudes=catalog_events.to_numpy(dtype=float),
        test_event_magnitudes=test_event_magnitudes,
    )
    return np.load(CACHE, allow_pickle=True)


def histogram_edges(values):
    lower = np.floor(np.nanmin(values) / HISTOGRAM_STEP) * HISTOGRAM_STEP
    upper = np.ceil(np.nanmax(values) / HISTOGRAM_STEP) * HISTOGRAM_STEP
    return np.arange(lower, upper + HISTOGRAM_STEP * 1.01, HISTOGRAM_STEP)


def save_histogram(values, title, filename):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(values, bins=histogram_edges(values), color="0.45", edgecolor="white", linewidth=0.7)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of earthquakes")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analysis_cache():
    if CACHE.exists():
        cache = np.load(CACHE, allow_pickle=True)
        if "analysis_level" in cache and str(cache["analysis_level"]) == "station_window_v1":
            return cache
    return build_cache()


def magnitude_edges(magnitudes):
    lower = np.floor(np.nanmin(magnitudes) / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH
    upper = np.ceil(np.nanmax(magnitudes) / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH
    edges = np.arange(lower, upper + MAGNITUDE_BIN_WIDTH * 1.01, MAGNITUDE_BIN_WIDTH)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def clustered_recall(frame, score_column, threshold, seed):
    grouped = frame.assign(detected=frame[score_column].ge(threshold)).groupby("source_id")["detected"].agg(["sum", "count"])
    point = grouped["sum"].sum() / grouped["count"].sum()
    rng = np.random.default_rng(seed)
    detected = grouped["sum"].to_numpy(dtype=float)
    totals = grouped["count"].to_numpy(dtype=float)
    bootstrap = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for index in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(grouped), len(grouped))
        bootstrap[index] = detected[selected].sum() / totals[selected].sum()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return point, lower, upper


def clustered_difference(frame, recovar_threshold, phasenet_threshold, seed):
    work = frame.assign(
        recovar_detected=frame["RECOVAR-INSTANCE"].ge(recovar_threshold),
        phasenet_detected=frame["PhaseNet-INSTANCE"].ge(phasenet_threshold),
    )
    grouped = work.groupby("source_id").agg(
        recovar_detected=("recovar_detected", "sum"),
        phasenet_detected=("phasenet_detected", "sum"),
        count=("source_id", "size"),
    )
    point = (grouped["recovar_detected"].sum() - grouped["phasenet_detected"].sum()) / grouped["count"].sum()
    rng = np.random.default_rng(seed)
    recovar = grouped["recovar_detected"].to_numpy(dtype=float)
    phasenet = grouped["phasenet_detected"].to_numpy(dtype=float)
    totals = grouped["count"].to_numpy(dtype=float)
    bootstrap = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for index in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(grouped), len(grouped))
        bootstrap[index] = (recovar[selected].sum() - phasenet[selected].sum()) / totals[selected].sum()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return point, lower, upper


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = analysis_cache()
    events = pd.DataFrame(
        {
            "source_id": cache["event_source_ids"].astype(str),
            "magnitude": cache["event_magnitudes"],
            "RECOVAR-INSTANCE": cache["recovar_event_scores"],
            "PhaseNet-INSTANCE": cache["phasenet_event_scores"],
        }
    )
    edges = magnitude_edges(events["magnitude"].to_numpy())
    models = {
        "RECOVAR-INSTANCE": cache["recovar_noise_scores"],
        "PhaseNet-INSTANCE": cache["phasenet_noise_scores"],
    }
    thresholds = {}
    actual_fprs = {}
    for model_name, noise_scores in models.items():
        thresholds[model_name] = {}
        actual_fprs[model_name] = {}
        for false_positive_rate in FALSE_POSITIVE_RATES:
            threshold = np.quantile(noise_scores, 1.0 - false_positive_rate, method="higher")
            thresholds[model_name][false_positive_rate] = threshold
            actual_fprs[model_name][false_positive_rate] = np.mean(noise_scores >= threshold)
    rows = []
    difference_rows = []
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        frame = events.loc[events["magnitude"].ge(left) & events["magnitude"].lt(right)]
        if frame.empty:
            continue
        for fpr_index, false_positive_rate in enumerate(FALSE_POSITIVE_RATES):
            for model_index, model_name in enumerate(models):
                recall, lower, upper = clustered_recall(
                    frame,
                    model_name,
                    thresholds[model_name][false_positive_rate],
                    10000 * bin_index + 100 * fpr_index + model_index,
                )
                rows.append(
                    {
                        "model": model_name,
                        "false_positive_rate_target": false_positive_rate,
                        "false_positive_rate_achieved": actual_fprs[model_name][false_positive_rate],
                        "threshold": thresholds[model_name][false_positive_rate],
                        "magnitude_left": left,
                        "magnitude_right": right,
                        "n_station_records": len(frame),
                        "n_source_events": frame["source_id"].nunique(),
                        "recall": recall,
                        "recall_ci_lower": lower,
                        "recall_ci_upper": upper,
                    }
                )
            difference, lower, upper = clustered_difference(
                frame,
                thresholds["RECOVAR-INSTANCE"][false_positive_rate],
                thresholds["PhaseNet-INSTANCE"][false_positive_rate],
                20000 * bin_index + fpr_index,
            )
            difference_rows.append(
                {
                    "false_positive_rate_target": false_positive_rate,
                    "magnitude_left": left,
                    "magnitude_right": right,
                    "n_station_records": len(frame),
                    "n_source_events": frame["source_id"].nunique(),
                    "recall_difference": difference,
                    "difference_ci_lower": lower,
                    "difference_ci_upper": upper,
                }
            )
    summary = pd.DataFrame(rows)
    differences = pd.DataFrame(difference_rows)
    summary.to_csv(OUTPUT / "instance_station_recall_by_magnitude.csv", index=False)
    differences.to_csv(OUTPUT / "instance_station_recall_difference_by_magnitude.csv", index=False)
    populated_bins = summary[["magnitude_left", "magnitude_right"]].drop_duplicates().reset_index(drop=True)
    labels = [f"[{row.magnitude_left:.1f}, {row.magnitude_right:.1f})" for row in populated_bins.itertuples()]
    x = np.arange(len(labels))
    colors = {"RECOVAR-INSTANCE": "#3b6ea8", "PhaseNet-INSTANCE": "#8b5ea7"}
    fig, axes = plt.subplots(len(FALSE_POSITIVE_RATES), 1, figsize=(11.0, 7.8), sharex=True)
    if len(FALSE_POSITIVE_RATES) == 1:
        axes = [axes]
    for axis, false_positive_rate in zip(axes, FALSE_POSITIVE_RATES):
        count_rows = summary.loc[summary["false_positive_rate_target"].eq(false_positive_rate) & summary["model"].eq("RECOVAR-INSTANCE")]
        count_axis = axis.twinx()
        count_axis.bar(x, count_rows["n_station_records"], width=0.82, color="0.85", edgecolor="none", label="Available station records")
        count_axis.set_ylabel("Available records", color="0.45")
        count_axis.tick_params(axis="y", colors="0.45")
        count_axis.spines["top"].set_visible(False)
        count_axis.set_zorder(0)
        axis.set_zorder(1)
        axis.patch.set_visible(False)
        for model_name in models:
            model_rows = summary.loc[summary["false_positive_rate_target"].eq(false_positive_rate) & summary["model"].eq(model_name)]
            values = model_rows["recall"].to_numpy()
            lower = values - model_rows["recall_ci_lower"].to_numpy()
            upper = model_rows["recall_ci_upper"].to_numpy() - values
            axis.errorbar(x, values, yerr=np.vstack([lower, upper]), marker="o", linewidth=1.7, capsize=2.5, color=colors[model_name], label=model_name, zorder=3)
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel("Station-level recall")
        axis.set_title(f"Global noise FPR = {100 * false_positive_rate:g}%")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Magnitude interval")
    fig.tight_layout()
    fig.savefig(OUTPUT / "instance_station_recall_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "instance_station_recall_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(len(FALSE_POSITIVE_RATES), 1, figsize=(11.0, 6.8), sharex=True)
    if len(FALSE_POSITIVE_RATES) == 1:
        axes = [axes]
    for axis, false_positive_rate in zip(axes, FALSE_POSITIVE_RATES):
        frame = differences.loc[differences["false_positive_rate_target"].eq(false_positive_rate)]
        values = frame["recall_difference"].to_numpy()
        lower = values - frame["difference_ci_lower"].to_numpy()
        upper = frame["difference_ci_upper"].to_numpy() - values
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        axis.errorbar(x, values, yerr=np.vstack([lower, upper]), marker="o", linewidth=1.7, capsize=2.5, color="#2f6b4f")
        axis.set_ylabel("Recall difference")
        axis.set_title(f"RECOVAR − PhaseNet at global noise FPR = {100 * false_positive_rate:g}%")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Magnitude interval")
    fig.tight_layout()
    fig.savefig(OUTPUT / "instance_station_recall_difference_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "instance_station_recall_difference_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_histogram(cache["full_magnitudes"], "INSTANCE catalog magnitude distribution", "instance_catalog_magnitude_histogram")
    save_histogram(cache["test_event_magnitudes"], "INSTANCE official test-set magnitude distribution", "instance_test_magnitude_histogram")
    print(OUTPUT / "instance_station_recall_by_magnitude.pdf")


if __name__ == "__main__":
    main()
