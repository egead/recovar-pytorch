import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility"))

from recovar_torch import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder
from seisbench_kfold_environment import SeisBenchKFoldEnvironment
from seisbench_kfold_trainer import get_dataset_time_window, load_seisbench_datasets


CHECKPOINT = Path("/home/ege/picovar/models/recovar_instance_seisbench_benchmark.pt")
CACHE = REPO / "instance_mag_cache.npz"
OUTPUT = REPO / "instance_magnitude_analysis"
N_BINS = 5
HISTOGRAM_STEP = 0.25


def magnitude_column(metadata):
    preferred = ["source_magnitude", "magnitude", "source_magnitude_ml", "source_magnitude_mw"]
    for name in preferred:
        if name in metadata.columns and pd.to_numeric(metadata[name], errors="coerce").notna().any():
            return name
    candidates = [name for name in metadata.columns if "magnitude" in name.lower() or name.lower() == "mag"]
    for name in candidates:
        if pd.to_numeric(metadata[name], errors="coerce").notna().any():
            return name
    raise KeyError(f"no numeric magnitude column found; available columns: {metadata.columns.tolist()}")


def score_recovar(generator):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"RECOVAR checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation = RepresentationLearningMultipleAutoencoder().to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    representation.load_state_dict(state)
    model = ClassifierMultipleAutoencoder(representation).to(device).eval()
    outputs = []
    with torch.inference_mode():
        for index in range(len(generator)):
            batch = torch.from_numpy(np.asarray(generator[index])).float().to(device)
            outputs.append(model(batch).cpu().numpy())
            print(f"RECOVAR completed:{index + 1}/{len(generator)}")
    return np.concatenate(outputs)


def create_cache():
    event_dataset, noise_dataset = load_seisbench_datasets("instance")
    environment = SeisBenchKFoldEnvironment(
        "instance",
        event_dataset=event_dataset,
        noise_dataset=noise_dataset,
        dataset_time_window=get_dataset_time_window("instance", event_dataset),
        apply_resampling=False,
    )
    _, _, test_metadata = environment.get_split_metadata(0)
    _, _, _, generator = environment.get_generators(0)
    scores = score_recovar(generator)
    test_metadata = test_metadata.reset_index(drop=True)
    if len(test_metadata) != len(scores):
        raise ValueError(f"test metadata has {len(test_metadata)} rows but RECOVAR produced {len(scores)} scores")
    source = event_dataset.metadata.reset_index(drop=True)
    mag_column = magnitude_column(source)
    source_magnitudes = pd.to_numeric(source[mag_column], errors="coerce")
    test_metadata["magnitude"] = np.nan
    event_rows = test_metadata["dataset_source"].eq("eq")
    event_indices = test_metadata.loc[event_rows, "sb_index"].astype(int)
    test_metadata.loc[event_rows, "magnitude"] = source_magnitudes.iloc[event_indices].to_numpy()
    test_metadata["score"] = scores
    detected_events = test_metadata.loc[
        test_metadata["label"].eq("eq") & test_metadata["magnitude"].notna()
    ]
    events = detected_events.groupby("source_id", as_index=False).agg(
        magnitude=("magnitude", "first"),
        score=("score", "max"),
    )
    noise_scores = test_metadata.loc[test_metadata["label"].eq("no"), "score"].to_numpy(dtype=float)
    full_magnitudes = source_magnitudes.dropna().to_numpy(dtype=float)
    if events.empty or noise_scores.size == 0:
        raise RuntimeError("INSTANCE test split did not produce both magnitude-bearing events and noise windows")
    np.savez_compressed(
        CACHE,
        event_magnitudes=events["magnitude"].to_numpy(dtype=float),
        event_scores=events["score"].to_numpy(dtype=float),
        noise_scores=noise_scores,
        full_magnitudes=full_magnitudes,
        magnitude_column=np.asarray(mag_column),
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
        for percentile, edge in zip([20, 40, 60, 80], percentile_edges[1:-1]):
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
    cache = np.load(CACHE) if CACHE.exists() else create_cache()
    magnitudes = cache["event_magnitudes"]
    event_scores = cache["event_scores"]
    noise_scores = cache["noise_scores"]
    edges = np.unique(np.quantile(magnitudes, np.linspace(0, 1, N_BINS + 1)))
    if len(edges) != N_BINS + 1:
        raise RuntimeError("magnitude quantiles collapsed because too few distinct magnitudes are present")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (magnitudes >= left) & (magnitudes < right)
        labels = np.concatenate([np.ones(selected.sum()), np.zeros(len(noise_scores))])
        values = np.concatenate([event_scores[selected], noise_scores])
        rows.append(
            {
                "model": "RECOVAR-INSTANCE",
                "percentile_left": index * 20,
                "percentile_right": (index + 1) * 20,
                "magnitude_left": left,
                "magnitude_right": right,
                "n_events": int(selected.sum()),
                "roc_auc": roc_auc_score(labels, values),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "instance_auc_by_magnitude.csv", index=False)
    labels = [
        f"{row.percentile_left:.0f}–{row.percentile_right:.0f}%\nM {row.magnitude_left:.2f}–{row.magnitude_right:.2f}\nn={row.n_events}"
        for row in summary.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(np.arange(len(summary)), summary["roc_auc"], marker="o", linewidth=1.8, color="#3b6ea8")
    ax.set_xticks(np.arange(len(summary)), labels)
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Test-event magnitude percentile and interval")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("INSTANCE detection performance by magnitude")
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT / "instance_auc_by_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "instance_auc_by_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_histogram(cache["full_magnitudes"], "INSTANCE catalog magnitude distribution", "instance_catalog_magnitude_histogram")
    save_histogram(magnitudes, "INSTANCE test-set magnitude distribution", "instance_test_magnitude_histogram", edges)
    print(OUTPUT / "instance_auc_by_magnitude.pdf")


if __name__ == "__main__":
    main()
