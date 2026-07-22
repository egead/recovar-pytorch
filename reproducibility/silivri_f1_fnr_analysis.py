import os
import sys
from pathlib import Path

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/mnt/second_drive/seisbench")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reproducibility"))

from silivri_all_mag_analysis import prepare_descriptors, magnitude_edges

# Load the normal (not butterworth) partial scores cache
PARTIAL_CACHE = REPO / "silivri_all_instance_scores_partial.npz"
OUTPUT = REPO / "silivri_f1_fnr_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

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

def main():
    if not PARTIAL_CACHE.exists():
        print(f"Error: Could not find {PARTIAL_CACHE}")
        print("Please ensure silivri_all_instance_scores_partial.npz is in the repository root.")
        return
        
    print("Loading descriptors (this parses Silivri metadata)...")
    descriptors, picks = prepare_descriptors()
    
    print("Loading raw scores from partial cache...")
    partial = np.load(PARTIAL_CACHE)
    recovar_scores = partial["recovar_scores"]
    phasenet_scores = partial["phasenet_scores"]
    
    if "states" in partial:
        states = partial["states"]
    else:
        # Fallback if states wasn't saved in this older cache
        states = np.where(np.isfinite(recovar_scores) & np.isfinite(phasenet_scores), 1, 0)
        
    valid = states == 1
    
    # Kind can be 'event' or 'noise'
    event_mask = descriptors["kind"].eq("event").to_numpy() & valid
    noise_mask = descriptors["kind"].eq("noise").to_numpy() & valid
    
    events_df = descriptors.loc[event_mask].copy()
    events_df["RECOVAR-INSTANCE"] = recovar_scores[event_mask]
    events_df["PhaseNet-INSTANCE"] = phasenet_scores[event_mask]
    
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
        print(f"  {m_name}: Max F1 threshold = {best_f1_thresh:.4f} (F1 = {best_f1_score:.4f})")
        
        for fnr in TARGET_FNRS:
            crit_name = f"FNR_{fnr}"
            thresh = find_fnr_threshold(valid_e, fnr)
            thresholds[m_name][crit_name] = thresh
            print(f"  {m_name}: {crit_name} threshold = {thresh:.4f}")

    edges = magnitude_edges(events_df["magnitude"].to_numpy())
    
    ds_summary_rows = []
    
    print("Evaluating recall for magnitude bins...")
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        frame = events_df.loc[events_df["magnitude"].ge(left) & events_df["magnitude"].lt(right)]
        if frame.empty:
            continue
        for crit_index, criterion in enumerate(CRITERIA):
            for m_index, m_name in enumerate(models_dict_events.keys()):
                thresh = thresholds[m_name][criterion]
                recall, lower, upper = clustered_recall(
                    frame,
                    m_name,
                    thresh,
                    10000 * bin_index + 100 * crit_index + m_index,
                )
                row = {
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
                }
                ds_summary_rows.append(row)
                
    ds_summary = pd.DataFrame(ds_summary_rows)
    ds_summary.to_csv(OUTPUT / "silivri_station_recall_by_magnitude_f1_fnr.csv", index=False)
    
    def plot_summary(df, suffix, max_mag=None):
        if df.empty:
            return
            
        if max_mag is not None:
            df = df[df["magnitude_left"] < max_mag].copy()
            if df.empty:
                return
            
        populated_bins = df[["magnitude_left", "magnitude_right"]].drop_duplicates().reset_index(drop=True)
        labels = [f"[{row.magnitude_left:.1f}, {row.magnitude_right:.1f})" for row in populated_bins.itertuples()]
        x = np.arange(len(labels))
        
        criteria = df["criterion"].unique()
        fig, axes = plt.subplots(len(criteria), 1, figsize=(12.0, 5.0 * len(criteria)), sharex=True)
        if len(criteria) == 1:
            axes = [axes]
            
        for axis, criterion in zip(axes, criteria):
            first_model = df["model"].iloc[0]
            count_rows = df.loc[df["criterion"].eq(criterion) & df["model"].eq(first_model)]
            
            count_axis = axis.twinx()
            count_axis.bar(x, count_rows["n_station_records"], width=0.82, color="0.90", edgecolor="none", label="Available records")
            count_axis.set_ylabel("Records Count", color="0.6")
            count_axis.tick_params(axis="y", colors="0.6")
            count_axis.spines["top"].set_visible(False)
            count_axis.set_zorder(0)
            
            axis.set_zorder(1)
            axis.patch.set_visible(False)
            
            min_recall_for_plot = 1.0
            
            for m_name in models_dict_events.keys():
                model_rows = df.loc[df["criterion"].eq(criterion) & df["model"].eq(m_name)]
                if model_rows.empty:
                    continue
                
                values = model_rows["recall"].to_numpy()
                lower = values - model_rows["recall_ci_lower"].to_numpy()
                upper = model_rows["recall_ci_upper"].to_numpy() - values
                
                valid_lower_bounds = model_rows["recall_ci_lower"].to_numpy()
                if len(valid_lower_bounds) > 0 and np.nanmin(valid_lower_bounds) < min_recall_for_plot:
                    min_recall_for_plot = np.nanmin(valid_lower_bounds)
                
                is_recovar = "RECOVAR" in m_name
                
                axis.errorbar(
                    x, values, yerr=np.vstack([lower, upper]),
                    marker="o" if is_recovar else "s",
                    linestyle="-" if is_recovar else "--",
                    linewidth=2.5 if is_recovar else 1.5,
                    capsize=3.0,
                    color="#2ca02c", # green for instance models
                    label=m_name,
                    zorder=3
                )
                
            y_bottom = max(0.0, min_recall_for_plot - 0.05)
            if np.isnan(y_bottom):
                y_bottom = 0.0
                
            axis.set_ylim(y_bottom, 1.05)
            axis.set_ylabel("Recall")
            axis.set_title(f"SILIVRI - Threshold Criterion = {criterion}")
            axis.grid(axis="y", color="0.85", linewidth=0.8, linestyle='--')
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=True, fontsize=10, loc="center left", bbox_to_anchor=(1.12, 0.5), shadow=True)
            
        axes[-1].set_xticks(x, labels, rotation=45, ha="right")
        axes[-1].set_xlabel("Magnitude interval")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"silivri_station_recall_by_magnitude_f1_fnr{suffix}.pdf", bbox_inches="tight")
        fig.savefig(OUTPUT / f"silivri_station_recall_by_magnitude_f1_fnr{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Generating plots...")
    plot_summary(ds_summary, suffix="")
    plot_summary(ds_summary, suffix="_under_mag3", max_mag=3.0)
    print("Done! Results in", OUTPUT)

if __name__ == "__main__":
    main()
