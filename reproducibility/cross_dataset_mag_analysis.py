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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from recovar_torch import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder

CHECKPOINT_CANDIDATES = [
    Path("/mnt/second_drive/ege/picovar/models"),
    Path.home() / "picovar/models",
]
OUTPUT = REPO / "cross_dataset_mag_analysis"
SAMPLING_RATE = 100
WINDOW_SAMPLES = 3000
BATCH_SIZE = 256
MAGNITUDE_BIN_WIDTH = 0.5
BOOTSTRAP_REPETITIONS = 1000
FALSE_POSITIVE_RATES = [0.001, 0.01]
PHASE_COLUMNS = [
    "trace_p_arrival_sample", "trace_pP_arrival_sample", "trace_P_arrival_sample",
    "trace_P1_arrival_sample", "trace_Pg_arrival_sample", "trace_Pn_arrival_sample",
    "trace_PmP_arrival_sample", "trace_pwP_arrival_sample", "trace_pwPm_arrival_sample",
    "trace_s_arrival_sample", "trace_S_arrival_sample", "trace_S1_arrival_sample",
    "trace_Sg_arrival_sample", "trace_SmS_arrival_sample", "trace_Sn_arrival_sample"
]

def get_checkpoint_path(model_name):
    for candidate in CHECKPOINT_CANDIDATES:
        path = candidate / f"recovar_{model_name}_seisbench_benchmark.pt"
        if path.exists():
            return path
    raise FileNotFoundError(f"RECOVAR checkpoint for {model_name} not found in {CHECKPOINT_CANDIDATES}")

def magnitude_column(metadata):
    preferred = ["source_magnitude", "source_magnitude_mw", "source_local_magnitude", "magnitude", "trace_magnitude"]
    for name in preferred:
        if name in metadata.columns and pd.to_numeric(metadata[name], errors="coerce").notna().any():
            return name
    candidates = [name for name in metadata.columns if "magnitude" in name.lower() or name.lower() == "mag"]
    for name in candidates:
        if pd.to_numeric(metadata[name], errors="coerce").notna().any():
            return name
    return None

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
    mag_col = magnitude_column(metadata)
    if mag_col:
        test["magnitude"] = pd.to_numeric(test[mag_col], errors="coerce")
    else:
        test["magnitude"] = np.nan
        
    if "source_id" not in test.columns:
        test["source_id"] = test.get("trace_name", test.index).astype(str)
    else:
        missing_source = test["source_id"].isna()
        if "trace_name" in test.columns:
            test.loc[missing_source, "source_id"] = test.loc[missing_source, "trace_name"].astype(str)
        test["source_id"] = test["source_id"].astype(str)
    return test.reset_index(drop=True)

def window_start(npts, first_arrival_val, descriptor_index):
    maximum = max(0, npts - WINDOW_SAMPLES)
    if np.isfinite(first_arrival_val):
        pre_seconds = 5.0 + ((descriptor_index * 7) % 101) / 10.0
        return int(np.clip(round(first_arrival_val - pre_seconds * SAMPLING_RATE), 0, maximum))
    if maximum == 0:
        return 0
    return int((descriptor_index * 2654435761) % (maximum + 1))

def recovar_preprocess(windows):
    # Removed Butterworth filter entirely, straight normalization
    output = windows.astype(np.float32)
    output -= output.mean(axis=2, keepdims=True)
    output /= 1e-12 + output.std(axis=2, keepdims=True)
    return np.transpose(output, (0, 2, 1))

def phasenet_preprocess(windows):
    output = windows.astype(np.float32)
    output -= output.mean(axis=2, keepdims=True)
    output /= 1e-12 + np.max(np.abs(output), axis=2, keepdims=True)
    return np.pad(output, ((0, 0), (0, 0), (0, 1)))

def load_models(model_name, device):
    representation = RepresentationLearningMultipleAutoencoder().to(device)
    state = torch.load(get_checkpoint_path(model_name), map_location=device, weights_only=False)
    representation.load_state_dict(state)
    recovar = ClassifierMultipleAutoencoder(representation).to(device).eval()
    try:
        phasenet = sbm.PhaseNet.from_pretrained(model_name).to(device).eval()
    except Exception as e:
        print(f"Failed to load PhaseNet for {model_name}, falling back to instance: {e}")
        phasenet = sbm.PhaseNet.from_pretrained("instance").to(device).eval()
    noise_index = list(phasenet.labels).index("N")
    return recovar, phasenet, noise_index

def score_test_set(dataset, descriptors, model_name):
    cache_path = OUTPUT / f"cache_{dataset.__class__.__name__}_{model_name}.npz"
    if cache_path.exists():
        partial = np.load(cache_path)
        if len(partial["recovar_scores"]) == len(descriptors):
            print(f"Loaded scores from {cache_path}")
            return partial["recovar_scores"], partial["phasenet_scores"]
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, model: {model_name}")
    recovar, phasenet, noise_index = load_models(model_name, device)
    
    recovar_scores = np.full(len(descriptors), np.nan, dtype=np.float32)
    phasenet_scores = np.full(len(descriptors), np.nan, dtype=np.float32)
    
    total_batches = int(np.ceil(len(descriptors) / BATCH_SIZE))
    with torch.inference_mode():
        for batch_index, begin in enumerate(range(0, len(descriptors), BATCH_SIZE)):
            end = min(begin + BATCH_SIZE, len(descriptors))
            windows = []
            for descriptor_index in range(begin, end):
                row = descriptors.iloc[descriptor_index]
                waveform, metadata = dataset.get_sample(int(row["dataset_index"]))
                waveform = np.asarray(waveform[:3], dtype=np.float32)
                if waveform.shape[0] != 3:
                    if waveform.shape[0] < 3:
                        waveform = np.pad(waveform, ((0, 3-waveform.shape[0]), (0, 0)))
                    else:
                        waveform = waveform[:3]
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
            
            if (batch_index + 1) % 10 == 0 or (batch_index + 1) == total_batches:
                print(f"Detection {model_name} completed:{batch_index + 1}/{total_batches}")
                
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, recovar_scores=recovar_scores, phasenet_scores=phasenet_scores)
    return recovar_scores, phasenet_scores

def magnitude_edges(magnitudes):
    magnitudes = magnitudes[~np.isnan(magnitudes)]
    if len(magnitudes) == 0:
        return np.array([0, 1])
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

def get_dataset_obj(dataset_name):
    if dataset_name == "ethz":
        return sbd.ETHZ(sampling_rate=SAMPLING_RATE, component_order="ZNE")
    elif dataset_name == "stead":
        return sbd.STEAD(sampling_rate=SAMPLING_RATE, component_order="ZNE")
    elif dataset_name == "instance":
        return sbd.InstanceCountsCombined(sampling_rate=SAMPLING_RATE, component_order="ZNE")
    raise ValueError(f"Unknown dataset {dataset_name}")

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATASETS = ["ethz", "stead", "instance"]
    MODELS = ["ethz", "stead", "instance"]
    
    all_summary_rows = []
    
    for ds_name in DATASETS:
        print(f"--- Processing Dataset: {ds_name} ---")
        dataset = get_dataset_obj(ds_name)
        descriptors = test_descriptors(dataset.metadata)
        print(f"Total test windows: {len(descriptors)}")
        
        events_df = descriptors.loc[descriptors["label"].eq("eq")].copy()
        noise_rows = descriptors["label"].eq("no").to_numpy()
        
        models_dict = {}
        thresholds = {}
        actual_fprs = {}
        
        for m_name in MODELS:
            recovar_scores, phasenet_scores = score_test_set(dataset, descriptors, m_name)
            
            recovar_col = f"RECOVAR-{m_name}"
            phasenet_col = f"PhaseNet-{m_name}"
            
            events_df[recovar_col] = recovar_scores[descriptors["label"].eq("eq").to_numpy()]
            events_df[phasenet_col] = phasenet_scores[descriptors["label"].eq("eq").to_numpy()]
            
            models_dict[recovar_col] = recovar_scores[noise_rows]
            models_dict[phasenet_col] = phasenet_scores[noise_rows]
            
        for model_name, noise_scores in models_dict.items():
            thresholds[model_name] = {}
            actual_fprs[model_name] = {}
            
            noise_scores_valid = noise_scores[np.isfinite(noise_scores)]
            if len(noise_scores_valid) == 0:
                print(f"Warning: No valid noise scores for {model_name} on {ds_name}. Using default 0.5")
                for fpr in FALSE_POSITIVE_RATES:
                    thresholds[model_name][fpr] = 0.5
                    actual_fprs[model_name][fpr] = 0.0
                continue
                
            for fpr in FALSE_POSITIVE_RATES:
                threshold = np.quantile(noise_scores_valid, 1.0 - fpr, method="higher")
                thresholds[model_name][fpr] = threshold
                actual_fprs[model_name][fpr] = np.mean(noise_scores_valid >= threshold)
                
        valid_mag_events = events_df.dropna(subset=["magnitude"])
        edges = magnitude_edges(valid_mag_events["magnitude"].to_numpy())
        
        ds_summary_rows = []
        
        for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            frame = valid_mag_events.loc[valid_mag_events["magnitude"].ge(left) & valid_mag_events["magnitude"].lt(right)]
            if frame.empty:
                continue
            for fpr_index, fpr in enumerate(FALSE_POSITIVE_RATES):
                for m_index, m_name in enumerate(models_dict.keys()):
                    recall, lower, upper = clustered_recall(
                        frame,
                        m_name,
                        thresholds[m_name][fpr],
                        10000 * bin_index + 100 * fpr_index + m_index,
                    )
                    row = {
                        "dataset": ds_name,
                        "model": m_name,
                        "false_positive_rate_target": fpr,
                        "false_positive_rate_achieved": actual_fprs[m_name][fpr],
                        "threshold": thresholds[m_name][fpr],
                        "magnitude_left": left,
                        "magnitude_right": right,
                        "n_station_records": len(frame),
                        "n_source_events": frame["source_id"].nunique(),
                        "recall": recall,
                        "recall_ci_lower": lower,
                        "recall_ci_upper": upper,
                    }
                    ds_summary_rows.append(row)
                    all_summary_rows.append(row)
                    
        ds_summary = pd.DataFrame(ds_summary_rows)
        ds_summary.to_csv(OUTPUT / f"{ds_name}_station_recall_by_magnitude.csv", index=False)
        
        if not ds_summary.empty:
            populated_bins = ds_summary[["magnitude_left", "magnitude_right"]].drop_duplicates().reset_index(drop=True)
            labels = [f"[{row.magnitude_left:.1f}, {row.magnitude_right:.1f})" for row in populated_bins.itertuples()]
            x = np.arange(len(labels))
            
            fig, axes = plt.subplots(len(FALSE_POSITIVE_RATES), 1, figsize=(12.0, 4.0 * len(FALSE_POSITIVE_RATES)), sharex=True)
            if len(FALSE_POSITIVE_RATES) == 1:
                axes = [axes]
                
            for axis, fpr in zip(axes, FALSE_POSITIVE_RATES):
                count_rows = ds_summary.loc[ds_summary["false_positive_rate_target"].eq(fpr) & ds_summary["model"].eq(f"RECOVAR-{MODELS[0]}")]
                count_axis = axis.twinx()
                count_axis.bar(x, count_rows["n_station_records"], width=0.82, color="0.85", edgecolor="none", label="Available records")
                count_axis.set_ylabel("Records", color="0.45")
                count_axis.tick_params(axis="y", colors="0.45")
                count_axis.spines["top"].set_visible(False)
                count_axis.set_zorder(0)
                axis.set_zorder(1)
                axis.patch.set_visible(False)
                
                for i, m_name in enumerate(models_dict.keys()):
                    model_rows = ds_summary.loc[ds_summary["false_positive_rate_target"].eq(fpr) & ds_summary["model"].eq(m_name)]
                    if model_rows.empty:
                        continue
                    values = model_rows["recall"].to_numpy()
                    lower = values - model_rows["recall_ci_lower"].to_numpy()
                    upper = model_rows["recall_ci_upper"].to_numpy() - values
                    axis.errorbar(x, values, yerr=np.vstack([lower, upper]), marker="o", linewidth=1.7, capsize=2.5, label=m_name, zorder=3)
                    
                axis.set_ylim(0.0, 1.03)
                axis.set_ylabel("Recall")
                axis.set_title(f"{ds_name.upper()} - FPR = {100 * fpr:g}%")
                axis.grid(axis="y", color="0.9", linewidth=0.6)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.1, 0.5))
                
            axes[-1].set_xticks(x, labels, rotation=45, ha="right")
            axes[-1].set_xlabel("Magnitude interval")
            fig.tight_layout()
            fig.savefig(OUTPUT / f"{ds_name}_station_recall_by_magnitude.pdf", bbox_inches="tight")
            fig.savefig(OUTPUT / f"{ds_name}_station_recall_by_magnitude.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    all_summary = pd.DataFrame(all_summary_rows)
    all_summary.to_csv(OUTPUT / "combined_station_recall_by_magnitude.csv", index=False)
    print("Done! Results in", OUTPUT)

if __name__ == "__main__":
    main()
