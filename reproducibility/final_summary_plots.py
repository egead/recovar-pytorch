import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/mnt/second_drive/seisbench")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility"))

from recovar_torch import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder
from silivri_all_mag_analysis import (
    prepare_descriptors, 
    station_directories, 
    waveform_index, 
    raw_window,
    magnitude_edges,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
    FILTER_SAMPLES,
    FILTER_CONTEXT_SAMPLES
)

# ----------------- INFERENCE -----------------
BATCH_SIZE = 32
def recovar_preprocess_normal(windows):
    center = windows[:, :, FILTER_CONTEXT_SAMPLES:FILTER_CONTEXT_SAMPLES + WINDOW_SAMPLES]
    output = center.astype(np.float32)
    output -= output.mean(axis=2, keepdims=True)
    output /= 1e-12 + output.std(axis=2, keepdims=True)
    return np.transpose(output, (0, 2, 1))

def load_recovar(model_path, device):
    print(f"DEBUG: Entering load_recovar for {model_path}", flush=True)
    representation = RepresentationLearningMultipleAutoencoder().to(device)
    print("DEBUG: Representation created.", flush=True)
    state = torch.load(model_path, map_location=device, weights_only=False)
    print("DEBUG: torch.load finished.", flush=True)
    representation.load_state_dict(state)
    print("DEBUG: load_state_dict finished.", flush=True)
    recovar = ClassifierMultipleAutoencoder(representation).to(device).eval()
    print("DEBUG: Classifier created.", flush=True)
    return recovar

def run_silivri_inference():
    UPDATED_CACHE = REPO / "silivri_all_instance_scores_partial_updated.npz"
    if UPDATED_CACHE.exists():
        chk = np.load(UPDATED_CACHE)
        if "recovar_dilation_scores" in chk and "recovar_nodilation_scores" in chk:
            print("Inference for Dilation/NoDilation already completed. Skipping.")
            return

    PARTIAL_CACHE = REPO / "silivri_all_instance_scores_partial.npz"
    if not PARTIAL_CACHE.exists():
        print("Missing partial cache! Cannot run inference without base cache.")
        return
        
    print("Loading original partial cache...")
    partial = np.load(PARTIAL_CACHE)
    recovar_scores = partial["recovar_scores"]
    phasenet_scores = partial["phasenet_scores"]
    
    out_dict = {
        "recovar_scores": recovar_scores,
        "phasenet_scores": phasenet_scores,
    }
    
    if "states" in partial:
        states = partial["states"]
        out_dict["states"] = states
    else:
        states = np.where(np.isfinite(recovar_scores) & np.isfinite(phasenet_scores), 1, 0)

    print("Loading descriptors...")
    descriptors, _ = prepare_descriptors()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_to_run = {
        "Dilation": "/mnt/second_drive/ege/picovar/models/silivri2019_dilation.pt",
        "NoDilation": "/mnt/second_drive/ege/picovar/models/silivri2019_nodilation.pt"
    }

    directories = station_directories()
    indices = {}
    stream_cache = OrderedDict()
    
    for name, path in models_to_run.items():
        if not os.path.exists(path):
            path = os.path.expanduser(f"~/picovar/models/silivri2019_{name.lower()}.pt")
            if not os.path.exists(path):
                print(f"Skipping {name}: model not found.")
                continue
                
        print(f"Running inference for {name}...")
        model = load_recovar(path, device)
        print("DEBUG: Model returned successfully. Allocating scores array...", flush=True)
        scores = np.full(len(descriptors), np.nan, dtype=np.float32)
        print("DEBUG: Scores array allocated.", flush=True)
        
        total_batches = int(np.ceil(len(descriptors) / BATCH_SIZE))
        with torch.inference_mode():
            for batch_index, begin in enumerate(range(0, len(descriptors), BATCH_SIZE)):
                print(f"DEBUG: Starting batch {batch_index + 1}/{total_batches}", flush=True)
                end = min(begin + BATCH_SIZE, len(descriptors))
                raw_windows = []
                valid_indices = []
                for descriptor_index, row in zip(range(begin, end), descriptors.iloc[begin:end].itertuples()):
                    if states[descriptor_index] <= 0: 
                        continue
                        
                    station = str(row.station_name).upper()
                    if station not in directories:
                        candidates = [p for k, p in directories.items() if station in k]
                        if not candidates: continue
                        directory = candidates[0]
                    else:
                        directory = directories[station]
                        
                    if station not in indices:
                        indices[station] = waveform_index(directory, station)
                        
                    try:
                        context_start = row.window_start - pd.Timedelta(seconds=FILTER_CONTEXT_SAMPLES / SAMPLING_RATE)
                        raw = raw_window(indices[station], station, context_start, stream_cache, FILTER_SAMPLES)
                        raw_windows.append(raw)
                        valid_indices.append(descriptor_index)
                    except Exception as e:
                        pass
                        
                print(f"DEBUG: Loaded {len(valid_indices)} windows. Preparing forward pass...", flush=True)
                if valid_indices:
                    raw_batch = np.stack(raw_windows)
                    recovar_input = torch.from_numpy(recovar_preprocess_normal(raw_batch)).to(device)
                    print(f"DEBUG: Input shape: {recovar_input.shape}. Passing to model...", flush=True)
                    scores[valid_indices] = model(recovar_input).cpu().numpy()
                    print(f"DEBUG: Forward pass complete.", flush=True)
                    
                if (batch_index + 1) % 5 == 0 or (batch_index + 1) == total_batches:
                    print(f"Batch {batch_index + 1}/{total_batches}", flush=True)
                    
        out_dict[f"recovar_{name.lower()}_scores"] = scores
        print(f"Finished {name}.")
        
    np.savez_compressed(UPDATED_CACHE, **out_dict)
    print(f"Saved new cache to {UPDATED_CACHE}")


# ----------------- ANALYSIS -----------------
TARGET_FNRS = [0.01, 0.05]
BOOTSTRAP_REPETITIONS = 1000

def find_best_f1_threshold(event_scores, noise_scores):
    if len(event_scores) == 0 or len(noise_scores) == 0:
        return 0.5, 0.0
    y_true = np.concatenate([np.ones_like(event_scores), np.zeros_like(noise_scores)])
    y_scores = np.concatenate([event_scores, noise_scores])
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / (tps + fps + 1e-12)
    recall = tps / (len(event_scores) + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    return y_scores[best_idx], f1[best_idx]

def find_fnr_threshold(event_scores, target_fnr):
    if len(event_scores) == 0:
        return 0.5
    return np.quantile(event_scores, target_fnr)

def clustered_recall(frame, score_column, threshold, seed):
    grouped = frame.assign(detected=frame[score_column].ge(threshold)).groupby("event_id")["detected"].agg(["sum", "count"])
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

def run_silivri_analysis():
    PARTIAL_CACHE = REPO / "silivri_all_instance_scores_partial.npz"
    OUTPUT = REPO / "silivri_f1_fnr_analysis"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PARTIAL_CACHE.exists():
        print(f"Error: Could not find {PARTIAL_CACHE}")
        return
        
    print("Loading descriptors for thresholding...")
    descriptors, picks = prepare_descriptors()
    partial = np.load(PARTIAL_CACHE)
    recovar_scores = partial["recovar_scores"]
    phasenet_scores = partial["phasenet_scores"]
    
    if "states" in partial:
        states = partial["states"]
    else:
        states = np.where(np.isfinite(recovar_scores) & np.isfinite(phasenet_scores), 1, 0)
        
    valid = states == 1
    event_mask = descriptors["kind"].eq("event").to_numpy() & valid
    noise_mask = descriptors["kind"].eq("noise").to_numpy() & valid
    
    events_df = descriptors.loc[event_mask].copy()
    models_dict_events = {
        "RECOVAR-INSTANCE": recovar_scores[event_mask],
        "PhaseNet-INSTANCE": phasenet_scores[event_mask]
    }
    models_dict_noise = {
        "RECOVAR-INSTANCE": recovar_scores[noise_mask],
        "PhaseNet-INSTANCE": phasenet_scores[noise_mask]
    }
    
    thresholds = {}
    CRITERIA = ["Max_F1"] + [f"FNR_{fnr}" for fnr in TARGET_FNRS]
    
    print("Calculating thresholds based on F1 and FNR criteria...")
    for m_name in models_dict_events.keys():
        thresholds[m_name] = {}
        e_scores = models_dict_events[m_name]
        n_scores = models_dict_noise[m_name]
        valid_e = e_scores[np.isfinite(e_scores)]
        valid_n = n_scores[np.isfinite(n_scores)]
        
        best_f1_thresh, best_f1_score = find_best_f1_threshold(valid_e, valid_n)
        thresholds[m_name]["Max_F1"] = best_f1_thresh
        for fnr in TARGET_FNRS:
            crit_name = f"FNR_{fnr}"
            thresh = find_fnr_threshold(valid_e, fnr)
            thresholds[m_name][crit_name] = thresh

    edges = magnitude_edges(events_df["magnitude"].to_numpy())
    for m_name, e_scores in models_dict_events.items():
        events_df[m_name] = e_scores
    ds_summary_rows = []
    
    print("Evaluating recall for magnitude bins...")
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        frame = events_df.loc[events_df["magnitude"].ge(left) & events_df["magnitude"].lt(right)]
        if frame.empty: continue
        for crit_index, criterion in enumerate(CRITERIA):
            for m_index, m_name in enumerate(models_dict_events.keys()):
                thresh = thresholds[m_name][criterion]
                recall, lower, upper = clustered_recall(
                    frame, m_name, thresh, 10000 * bin_index + 100 * crit_index + m_index
                )
                ds_summary_rows.append({
                    "dataset": "silivri",
                    "model": m_name,
                    "criterion": criterion,
                    "threshold": thresh,
                    "magnitude_left": left,
                    "magnitude_right": right,
                    "n_station_records": len(frame),
                    "n_catalog_events": frame["event_id"].nunique(),
                    "recall": recall,
                    "recall_ci_lower": lower,
                    "recall_ci_upper": upper,
                })
                
    ds_summary = pd.DataFrame(ds_summary_rows)
    ds_summary.to_csv(OUTPUT / "silivri_station_recall_by_magnitude_f1_fnr.csv", index=False)


# ----------------- PLOTTING -----------------
OUTPUT_DIR = REPO / "final_summary_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_SNR = REPO / "cross_dataset_snr_f1_fnr_analysis/combined_station_recall_by_snr_f1_fnr.csv"
CSV_MAG = REPO / "cross_dataset_f1_fnr_analysis/combined_station_recall_by_magnitude_f1_fnr.csv"
CSV_SILIVRI = REPO / "silivri_f1_fnr_analysis/silivri_station_recall_by_magnitude_f1_fnr.csv"

def plot_2x2(csv_path, x_col, out_name, min_x=None, max_x=None):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = df[df["criterion"] == "FNR_0.01"].copy()
    if min_x is not None: df = df[df[f"{x_col}_left"] >= min_x]
    if max_x is not None: df = df[df[f"{x_col}_right"] <= max_x]
        
    datasets = ["stead", "instance"]
    model_pairs = [["PhaseNet-instance", "RECOVAR-instance"], ["PhaseNet-stead", "RECOVAR-stead"]]
    col_titles = ["Instance Models", "STEAD Models"]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    for row_idx, ds in enumerate(datasets):
        for col_idx, pair in enumerate(model_pairs):
            ax = axes[row_idx, col_idx]
            plot_df = df[df["dataset"] == ds].copy()
            if plot_df.empty: continue
            
            populated_bins = plot_df[[f"{x_col}_left", f"{x_col}_right"]].drop_duplicates().sort_values(f"{x_col}_left").reset_index(drop=True)
            labels = [f"[{row[f'{x_col}_left']:.1f}, {row[f'{x_col}_right']:.1f})" for _, row in populated_bins.iterrows()]
            x = np.arange(len(labels))
            
            first_model = pair[0]
            count_rows = plot_df[plot_df["model"] == first_model].set_index(f"{x_col}_left").reindex(populated_bins[f"{x_col}_left"]).fillna(0)
            
            count_ax = ax.twinx()
            count_ax.bar(x, count_rows["n_station_records"], width=0.8, color="0.9", edgecolor="none", label="Records")
            count_ax.set_ylabel("Records", color="0.6")
            count_ax.tick_params(axis="y", colors="0.6")
            count_ax.spines["top"].set_visible(False)
            count_ax.set_zorder(0)
            
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            min_recall = 1.0
            
            width = 0.35
            offset = width / 2
            for m_idx, m_name in enumerate(pair):
                m_df = plot_df[plot_df["model"] == m_name].set_index(f"{x_col}_left").reindex(populated_bins[f"{x_col}_left"])
                vals = m_df["recall"].to_numpy(dtype=float)
                valid_mask = ~np.isnan(vals)
                if not valid_mask.any(): continue
                    
                lower = vals - m_df["recall_ci_lower"].to_numpy(dtype=float)
                upper = m_df["recall_ci_upper"].to_numpy(dtype=float) - vals
                valid_lower = m_df["recall_ci_lower"].dropna()
                if not valid_lower.empty and valid_lower.min() < min_recall:
                    min_recall = valid_lower.min()
                    
                is_recovar = "RECOVAR" in m_name
                x_pos = x[valid_mask] - offset if m_idx == 0 else x[valid_mask] + offset
                ax.bar(
                    x_pos, vals[valid_mask], width=width,
                    yerr=[lower[valid_mask], upper[valid_mask]],
                    color="#1f77b4" if is_recovar else "#ff7f0e",
                    label=m_name, zorder=3, alpha=0.9, capsize=3.0
                )
                
            y_bot = max(0.0, min_recall - 0.05)
            if np.isnan(y_bot): y_bot = 0.0
            ax.set_ylim(y_bot, 1.05)
            ax.set_ylabel("Recall")
            ax.set_title(f"Dataset: {ds.upper()} | {col_titles[col_idx]}")
            ax.grid(axis="y", color="0.85", linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=True, fontsize=10, loc="lower right", shadow=True)
            
            if row_idx == 1:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_xlabel(f"{'SNR (dB)' if x_col == 'snr' else 'Magnitude'}")
                
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_silivri(csv_path, out_name):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = df[df["criterion"] == "FNR_0.01"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    configs = [{"ax": axes[0], "title": "Full Magnitude Range", "max_mag": None},
               {"ax": axes[1], "title": "Zoomed (< Mag 3.0)", "max_mag": 3.0}]
    
    models = df["model"].unique().tolist()
    
    for cfg in configs:
        ax = cfg["ax"]
        plot_df = df.copy()
        if cfg["max_mag"] is not None:
            plot_df = plot_df[plot_df["magnitude_right"] <= cfg["max_mag"] + 0.01]
            
        if plot_df.empty: continue
            
        populated_bins = plot_df[["magnitude_left", "magnitude_right"]].drop_duplicates().sort_values("magnitude_left").reset_index(drop=True)
        labels = [f"[{row['magnitude_left']:.1f}, {row['magnitude_right']:.1f})" for _, row in populated_bins.iterrows()]
        x = np.arange(len(labels))
        
        first_model = models[0]
        count_rows = plot_df[plot_df["model"] == first_model].set_index("magnitude_left").reindex(populated_bins["magnitude_left"]).fillna(0)
        
        count_ax = ax.twinx()
        count_ax.bar(x, count_rows["n_station_records"], width=0.8, color="0.9", edgecolor="none", label="Records")
        count_ax.set_ylabel("Records Count", color="0.6")
        count_ax.tick_params(axis="y", colors="0.6")
        count_ax.spines["top"].set_visible(False)
        count_ax.set_zorder(0)
        
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        min_recall = 1.0
        
        width = 0.35
        offset = width / 2
        for m_idx, m_name in enumerate(models):
            m_df = plot_df[plot_df["model"] == m_name].set_index("magnitude_left").reindex(populated_bins["magnitude_left"])
            vals = m_df["recall"].to_numpy(dtype=float)
            valid_mask = ~np.isnan(vals)
            if not valid_mask.any(): continue
                
            lower = vals - m_df["recall_ci_lower"].to_numpy(dtype=float)
            upper = m_df["recall_ci_upper"].to_numpy(dtype=float) - vals
            valid_lower = m_df["recall_ci_lower"].dropna()
            if not valid_lower.empty and valid_lower.min() < min_recall:
                min_recall = valid_lower.min()
                
            is_recovar = "RECOVAR" in m_name
            color = "#1f77b4" if is_recovar else "#ff7f0e"
            
            x_pos = x[valid_mask] - offset if m_idx == 0 else x[valid_mask] + offset
            ax.bar(
                x_pos, vals[valid_mask], width=width,
                yerr=[lower[valid_mask], upper[valid_mask]],
                color=color,
                label=m_name, zorder=3, alpha=0.9, capsize=3.0
            )
            
        y_bot = max(0.0, min_recall - 0.05)
        if np.isnan(y_bot): y_bot = 0.0
        ax.set_ylim(y_bot, 1.05)
        ax.set_ylabel("Recall")
        ax.set_title(f"SILIVRI | {cfg['title']}")
        ax.grid(axis="y", color="0.85", linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, fontsize=10, loc="lower right", shadow=True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Magnitude")
            
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    print("\n--- Step 1: Computing 1% FNR Thresholds for Silivri ---")
    run_silivri_analysis()

    print("\n--- Step 2: Generating Final Summary Plots ---")
    print("Generating SNR 2x2 Plot (>= 0 dB)...")
    plot_2x2(CSV_SNR, "snr", "snr_2x2_stead_instance.png", min_x=0.0)
    
    print("Generating Magnitude 2x2 Plot...")
    plot_2x2(CSV_MAG, "magnitude", "magnitude_2x2_stead_instance.png")
    
    print("Generating Silivri 1x2 Plot...")
    plot_silivri(CSV_SILIVRI, "magnitude_1x2_silivri.png")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
