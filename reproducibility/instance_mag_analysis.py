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
from sklearn.metrics import roc_auc_score


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
N_BINS = 10
HISTOGRAM_STEP = 0.25
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
    events["RECOVAR-INSTANCE"] = recovar_scores[event_rows]
    events["PhaseNet-INSTANCE"] = phasenet_scores[event_rows]
    events = events.groupby("source_id", as_index=False).agg(
        magnitude=("magnitude", "first"),
        **{
            "RECOVAR-INSTANCE": ("RECOVAR-INSTANCE", "max"),
            "PhaseNet-INSTANCE": ("PhaseNet-INSTANCE", "max"),
        },
    )
    noise_rows = descriptors["label"].eq("no").to_numpy()
    full_magnitudes = pd.to_numeric(dataset.metadata[magnitude_column(dataset.metadata)], errors="coerce").dropna().to_numpy()
    np.savez_compressed(
        CACHE,
        event_magnitudes=events["magnitude"].to_numpy(dtype=float),
        recovar_event_scores=events["RECOVAR-INSTANCE"].to_numpy(dtype=float),
        phasenet_event_scores=events["PhaseNet-INSTANCE"].to_numpy(dtype=float),
        recovar_noise_scores=recovar_scores[noise_rows],
        phasenet_noise_scores=phasenet_scores[noise_rows],
        full_magnitudes=full_magnitudes,
    )
    return np.load(CACHE)


def histogram_edges(values):
    lower = np.floor(np.nanmin(values) / HISTOGRAM_STEP) * HISTOGRAM_STEP
    upper = np.ceil(np.nanmax(values) / HISTOGRAM_STEP) * HISTOGRAM_STEP
    return np.arange(lower, upper + HISTOGRAM_STEP * 1.01, HISTOGRAM_STEP)


def save_histogram(values, title, filename, percentile_edges=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(values, bins=histogram_edges(values), color="0.45", edgecolor="white", linewidth=0.7)
    if percentile_edges is not None:
        percentiles = np.arange(1, len(percentile_edges) - 1) * 100 / (len(percentile_edges) - 1)
        for percentile, edge in zip(percentiles, percentile_edges[1:-1]):
            ax.axvline(edge, color="0.2", linestyle="--", linewidth=0.9, label=f"{percentile}th percentile")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of earthquakes")
    ax.set_title(title)
    if percentile_edges is not None:
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE) if CACHE.exists() else build_cache()
    magnitudes = cache["event_magnitudes"]
    edges = np.unique(np.quantile(magnitudes, np.linspace(0, 1, N_BINS + 1)))
    if len(edges) != N_BINS + 1:
        raise RuntimeError("magnitude quantiles collapsed because too few distinct magnitudes are present")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    models = {
        "RECOVAR-INSTANCE": (cache["recovar_event_scores"], cache["recovar_noise_scores"]),
        "PhaseNet-INSTANCE": (cache["phasenet_event_scores"], cache["phasenet_noise_scores"]),
    }
    percentile_step = 100 / N_BINS
    rows = []
    for model_name, (event_scores, noise_scores) in models.items():
        thresholds = {
            0.001: np.quantile(noise_scores, 0.999, method="higher"),
            0.01: np.quantile(noise_scores, 0.99, method="higher"),
        }
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            selected = (magnitudes >= left) & (magnitudes < right)
            labels = np.concatenate([np.ones(selected.sum()), np.zeros(len(noise_scores))])
            values = np.concatenate([event_scores[selected], noise_scores])
            rows.append(
                {
                    "model": model_name,
                    "percentile_left": index * percentile_step,
                    "percentile_right": (index + 1) * percentile_step,
                    "magnitude_left": left,
                    "magnitude_right": right,
                    "n_events": int(selected.sum()),
                    "roc_auc": roc_auc_score(labels, values),
                    "threshold_at_fpr_0_001": thresholds[0.001],
                    "actual_fpr_0_001": np.mean(noise_scores >= thresholds[0.001]),
                    "recall_at_fpr_0_001": np.mean(event_scores[selected] >= thresholds[0.001]),
                    "threshold_at_fpr_0_01": thresholds[0.01],
                    "actual_fpr_0_01": np.mean(noise_scores >= thresholds[0.01]),
                    "recall_at_fpr_0_01": np.mean(event_scores[selected] >= thresholds[0.01]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "instance_auc_by_magnitude.csv", index=False)
    labels = [
        f"{index * percentile_step:.0f}–{(index + 1) * percentile_step:.0f}%\nM {left:.2f}–{right:.2f}"
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    colors = {"RECOVAR-INSTANCE": "#3b6ea8", "PhaseNet-INSTANCE": "#8b5ea7"}
    x = np.arange(len(labels))
    width = 0.8 / len(models)
    for model_index, model_name in enumerate(models):
        model_rows = summary.loc[summary["model"].eq(model_name)]
        offset = (model_index - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            model_rows["roc_auc"].to_numpy(),
            width=width,
            color=colors[model_name],
            edgecolor="0.2",
            linewidth=0.5,
            label=model_name,
        )
        for bar, count in zip(bars, model_rows["n_events"].to_numpy()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"n={count}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0.45, 1.08)
    ax.set_xlabel("Magnitude interval")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("INSTANCE test-set detection")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.4, labelspacing=0.45)
    fig.tight_layout()
    fig.savefig(OUTPUT / "instance_auc_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "instance_auc_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    recall_series = [
        ("RECOVAR-INSTANCE", "recall_at_fpr_0_001", "RECOVAR, FPR 0.1%", "#3b6ea8"),
        ("PhaseNet-INSTANCE", "recall_at_fpr_0_001", "PhaseNet, FPR 0.1%", "#8b5ea7"),
        ("RECOVAR-INSTANCE", "recall_at_fpr_0_01", "RECOVAR, FPR 1%", "#86afd3"),
        ("PhaseNet-INSTANCE", "recall_at_fpr_0_01", "PhaseNet, FPR 1%", "#bea6cf"),
    ]
    width = 0.8 / len(recall_series)
    for series_index, (model_name, column, legend, color) in enumerate(recall_series):
        values = summary.loc[summary["model"].eq(model_name), column].to_numpy()
        offset = (series_index - (len(recall_series) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, color=color, edgecolor="0.2", linewidth=0.5, label=legend)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Magnitude interval")
    ax.set_ylabel("Recall")
    ax.set_title("INSTANCE test-set recall at fixed false-positive rates")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "instance_recall_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "instance_recall_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_histogram(cache["full_magnitudes"], "INSTANCE catalog magnitude distribution", "instance_catalog_magnitude_histogram")
    save_histogram(magnitudes, "INSTANCE official test-set magnitude distribution", "instance_test_magnitude_histogram", edges)
    print(OUTPUT / "instance_auc_by_magnitude.pdf")


if __name__ == "__main__":
    main()
