import os
import sys
from pathlib import Path

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/mnt/second_drive/seisbench")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seisbench.data as sbd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Read the cached outputs
CACHE_DIR = REPO / "cross_dataset_mag_analysis"
OUTPUT = REPO / "cross_dataset_snr_f1_fnr_analysis"

SAMPLING_RATE = 100
SNR_BIN_WIDTH = 5.0
BOOTSTRAP_REPETITIONS = 1000
TARGET_FNRS = [0.01, 0.05] 

PHASE_COLUMNS = [
    "trace_p_arrival_sample", "trace_pP_arrival_sample", "trace_P_arrival_sample",
    "trace_P1_arrival_sample", "trace_Pg_arrival_sample", "trace_Pn_arrival_sample",
    "trace_PmP_arrival_sample", "trace_pwP_arrival_sample", "trace_pwPm_arrival_sample",
    "trace_s_arrival_sample", "trace_S_arrival_sample", "trace_S1_arrival_sample",
    "trace_Sg_arrival_sample", "trace_SmS_arrival_sample", "trace_Sn_arrival_sample"
]

def extract_snr(metadata, dataset_name):
    df = metadata.copy()
    if dataset_name == "stead" and "snr_db" in df.columns:
        def parse_stead(x):
            if pd.isna(x): return np.nan, np.nan, np.nan
            try:
                s = str(x).replace("[", "").replace("]", "")
                parts = [p.strip() for p in s.split() if p.strip()]
                return float(parts[0]), float(parts[1]), float(parts[2])
            except:
                return np.nan, np.nan, np.nan
        
        parsed = df["snr_db"].apply(parse_stead)
        df["trace_E_snr_db"] = [p[0] for p in parsed]
        df["trace_N_snr_db"] = [p[1] for p in parsed]
        df["trace_Z_snr_db"] = [p[2] for p in parsed]
        
    if "trace_E_snr_db" in df.columns and "trace_N_snr_db" in df.columns and "trace_Z_snr_db" in df.columns:
        E = pd.to_numeric(df["trace_E_snr_db"], errors="coerce")
        N = pd.to_numeric(df["trace_N_snr_db"], errors="coerce")
        Z = pd.to_numeric(df["trace_Z_snr_db"], errors="coerce")
        
        # Avoid overflow warnings
        with np.errstate(over='ignore'):
            combined = 10 * np.log10(10**(E/10) + 10**(N/10) + 10**(Z/10))
        return combined
    else:
        return pd.Series(np.nan, index=df.index)

def test_descriptors(metadata, dataset_name):
    phase_columns = [name for name in PHASE_COLUMNS if name in metadata.columns]
    if "split" not in metadata.columns:
        raise KeyError("metadata has no split column")
    test = metadata.loc[metadata["split"].eq("test")].copy()
    if phase_columns:
        arrivals = test[phase_columns].apply(pd.to_numeric, errors="coerce")
        test["first_arrival"] = arrivals.min(axis=1, skipna=True)
    else:
        test["first_arrival"] = np.nan
    
    if "trace_category" in test.columns:
        test["label"] = np.where(test["trace_category"].str.contains("earthquake", na=False), "eq", "no")
    else:
        test["label"] = np.where(test["first_arrival"].notna(), "eq", "no")
        
    test["dataset_index"] = test.index.astype(int)
    test["snr"] = extract_snr(test, dataset_name)
        
    if "source_id" not in test.columns:
        test["source_id"] = test.get("trace_name", test.index).astype(str)
    else:
        missing_source = test["source_id"].isna()
        if "trace_name" in test.columns:
            test.loc[missing_source, "source_id"] = test.loc[missing_source, "trace_name"].astype(str)
        test["source_id"] = test["source_id"].astype(str)
    return test.reset_index(drop=True)

def snr_edges(snrs):
    snrs = snrs[~np.isnan(snrs)]
    if len(snrs) == 0:
        return np.array([0, 1])
    lower = np.floor(np.nanmin(snrs) / SNR_BIN_WIDTH) * SNR_BIN_WIDTH
    upper = np.ceil(np.nanmax(snrs) / SNR_BIN_WIDTH) * SNR_BIN_WIDTH
    edges = np.arange(lower, upper + SNR_BIN_WIDTH * 1.01, SNR_BIN_WIDTH)
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

def get_dataset_obj(dataset_name):
    if dataset_name == "stead":
        return sbd.STEAD(sampling_rate=SAMPLING_RATE, component_order="ZNE")
    elif dataset_name == "instance":
        return sbd.InstanceCountsCombined(sampling_rate=SAMPLING_RATE, component_order="ZNE")
    raise ValueError(f"Unknown dataset {dataset_name}")

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

def plot_summary(ds_summary, ds_name, models_dict_events, output_dir, max_snr=None):
    if ds_summary.empty:
        return
        
    if max_snr is not None:
        ds_summary = ds_summary[ds_summary["snr_left"] < max_snr].copy()
        if ds_summary.empty:
            return
        suffix = f"_under_snr{int(max_snr)}"
    else:
        suffix = ""
        
    populated_bins = ds_summary[["snr_left", "snr_right"]].drop_duplicates().reset_index(drop=True)
    labels = [f"[{row.snr_left:.1f}, {row.snr_right:.1f})" for row in populated_bins.itertuples()]
    x = np.arange(len(labels))
    
    criteria = ds_summary["criterion"].unique()
    fig, axes = plt.subplots(len(criteria), 1, figsize=(12.0, 5.0 * len(criteria)), sharex=True)
    if len(criteria) == 1:
        axes = [axes]
        
    colors = {
        'stead': '#ff7f0e', # orange
        'instance': '#2ca02c' # green
    }
    
    for axis, criterion in zip(axes, criteria):
        first_model = ds_summary["model"].iloc[0]
        count_rows = ds_summary.loc[ds_summary["criterion"].eq(criterion) & ds_summary["model"].eq(first_model)]
        
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
            model_rows = ds_summary.loc[ds_summary["criterion"].eq(criterion) & ds_summary["model"].eq(m_name)]
            if model_rows.empty:
                continue
            
            values = model_rows["recall"].to_numpy()
            lower = values - model_rows["recall_ci_lower"].to_numpy()
            upper = model_rows["recall_ci_upper"].to_numpy() - values
            
            valid_lower_bounds = model_rows["recall_ci_lower"].to_numpy()
            if len(valid_lower_bounds) > 0 and np.nanmin(valid_lower_bounds) < min_recall_for_plot:
                min_recall_for_plot = np.nanmin(valid_lower_bounds)
            
            base_model = m_name.split('-')[1]
            is_recovar = "RECOVAR" in m_name
            
            axis.errorbar(
                x, values, yerr=np.vstack([lower, upper]),
                marker="o" if is_recovar else "s",
                linestyle="-" if is_recovar else "--",
                linewidth=2.5 if is_recovar else 1.5,
                capsize=3.0,
                color=colors.get(base_model, '#333333'),
                label=m_name,
                zorder=3
            )
            
        y_bottom = max(0.0, min_recall_for_plot - 0.05)
        if np.isnan(y_bottom):
            y_bottom = 0.0
            
        axis.set_ylim(y_bottom, 1.05)
        axis.set_ylabel("Recall")
        axis.set_title(f"{ds_name.upper()} - Threshold Criterion = {criterion}")
        axis.grid(axis="y", color="0.85", linewidth=0.8, linestyle='--')
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=True, fontsize=10, loc="center left", bbox_to_anchor=(1.12, 0.5), shadow=True)
        
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1].set_xlabel("SNR (dB)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ds_name}_station_recall_by_snr_f1_fnr{suffix}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{ds_name}_station_recall_by_snr_f1_fnr{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATASETS = ["stead", "instance"]
    MODELS = ["stead", "instance"]
    CRITERIA = ["Max_F1"] + [f"FNR_{fnr}" for fnr in TARGET_FNRS]
    
    all_summary_rows = []
    
    for ds_name in DATASETS:
        print(f"--- Processing Dataset for SNR: {ds_name} ---")
        dataset = get_dataset_obj(ds_name)
        descriptors = test_descriptors(dataset.metadata, ds_name)
        
        events_df = descriptors.loc[descriptors["label"].eq("eq")].copy()
        noise_rows = descriptors["label"].eq("no").to_numpy()
        event_rows = descriptors["label"].eq("eq").to_numpy()
        
        models_dict_events = {}
        models_dict_noise = {}
        thresholds = {}
        
        for m_name in MODELS:
            cache_path = CACHE_DIR / f"cache_{dataset.__class__.__name__}_{m_name}.npz"
            if not cache_path.exists():
                print(f"Cache missing for {ds_name} / {m_name}, skipping.")
                continue
                
            partial = np.load(cache_path)
            recovar_scores = partial["recovar_scores"]
            phasenet_scores = partial["phasenet_scores"]
            
            if len(recovar_scores) != len(descriptors):
                print(f"Mismatch in cache length for {ds_name} / {m_name}. Expected {len(descriptors)}, got {len(recovar_scores)}. Skipping.")
                continue
            
            recovar_col = f"RECOVAR-{m_name}"
            phasenet_col = f"PhaseNet-{m_name}"
            
            events_df[recovar_col] = recovar_scores[event_rows]
            events_df[phasenet_col] = phasenet_scores[event_rows]
            models_dict_events[recovar_col] = recovar_scores[event_rows]
            models_dict_events[phasenet_col] = phasenet_scores[event_rows]
            
            models_dict_noise[recovar_col] = recovar_scores[noise_rows]
            models_dict_noise[phasenet_col] = phasenet_scores[noise_rows]
            
        for model_name in models_dict_events.keys():
            thresholds[model_name] = {}
            e_scores = models_dict_events[model_name]
            n_scores = models_dict_noise[model_name]
            
            valid_e = e_scores[np.isfinite(e_scores)]
            valid_n = n_scores[np.isfinite(n_scores)]
            
            best_f1_thresh, best_f1_score = find_best_f1_threshold(valid_e, valid_n)
            thresholds[model_name]["Max_F1"] = best_f1_thresh
            print(f"  {model_name}: Max F1 threshold = {best_f1_thresh:.4f}")
            
            for fnr in TARGET_FNRS:
                crit_name = f"FNR_{fnr}"
                thresh = find_fnr_threshold(valid_e, fnr)
                thresholds[model_name][crit_name] = thresh
                print(f"  {model_name}: {crit_name} threshold = {thresh:.4f}")

        valid_snr_events = events_df.dropna(subset=["snr"])
        edges = snr_edges(valid_snr_events["snr"].to_numpy())
        
        ds_summary_rows = []
        for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            frame = valid_snr_events.loc[valid_snr_events["snr"].ge(left) & valid_snr_events["snr"].lt(right)]
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
                        "dataset": ds_name,
                        "model": m_name,
                        "criterion": criterion,
                        "threshold": thresh,
                        "snr_left": left,
                        "snr_right": right,
                        "n_station_records": len(frame),
                        "n_source_events": frame["source_id"].nunique(),
                        "recall": recall,
                        "recall_ci_lower": lower,
                        "recall_ci_upper": upper,
                    }
                    ds_summary_rows.append(row)
                    all_summary_rows.append(row)
                    
        ds_summary = pd.DataFrame(ds_summary_rows)
        if ds_summary.empty:
            continue
            
        ds_summary.to_csv(OUTPUT / f"{ds_name}_station_recall_by_snr_f1_fnr.csv", index=False)
        plot_summary(ds_summary, ds_name, models_dict_events, OUTPUT, max_snr=None)
        # For SNR, maybe a plot zooming in on lower SNRs (e.g. < 10 dB)
        plot_summary(ds_summary, ds_name, models_dict_events, OUTPUT, max_snr=10.0)

    all_summary = pd.DataFrame(all_summary_rows)
    all_summary.to_csv(OUTPUT / "combined_station_recall_by_snr_f1_fnr.csv", index=False)
    print("Done! Results in", OUTPUT)

if __name__ == "__main__":
    main()
