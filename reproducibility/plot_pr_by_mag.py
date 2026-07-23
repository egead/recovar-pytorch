import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from reproducibility.cross_dataset_mag_analysis import get_dataset_obj, test_descriptors, magnitude_edges
from sklearn.metrics import precision_recall_curve, auc

OUTPUT_DIR = REPO / "cross_dataset_mag_analysis"

def main():
    DATASETS = ["stead", "instance", "silivri"]
    
    for ds_name in DATASETS:
        events_df = None
        if ds_name in ["stead", "instance"]:
            dataset = get_dataset_obj(ds_name)
            descriptors = test_descriptors(dataset.metadata)
            
            events_df = descriptors.loc[descriptors["label"].eq("eq")].copy()
            noise_rows = descriptors["label"].eq("no").to_numpy()
            event_rows = descriptors["label"].eq("eq").to_numpy()
            
            m_name = ds_name # Evaluate each dataset with its matching model
            cache_path = OUTPUT_DIR / f"cache_{dataset.__class__.__name__}_{m_name}.npz"
            if not cache_path.exists():
                print(f"Cache not found: {cache_path}")
                continue
                
            scores = np.load(cache_path)
            recovar_scores = scores["recovar_scores"]
            phasenet_scores = scores["phasenet_scores"]
            
            noise_recovar = recovar_scores[noise_rows]
            noise_phasenet = phasenet_scores[noise_rows]
            
            events_df["recovar"] = recovar_scores[event_rows]
            events_df["phasenet"] = phasenet_scores[event_rows]
            
            edges = magnitude_edges(events_df["magnitude"].to_numpy())
        elif ds_name == "silivri":
            cache_path = REPO / "silivri_all_instance_butterworth_context_mag_cache_new_train.npz"
            if not cache_path.exists():
                print(f"Cache not found: {cache_path}")
                continue
            scores = np.load(cache_path)
            events_df = pd.DataFrame({
                "magnitude": scores["event_magnitudes"],
                "recovar": scores["recovar_event_scores"],
                "phasenet": scores["phasenet_event_scores"]
            })
            noise_recovar = scores["recovar_noise_validation"]
            noise_phasenet = scores["phasenet_noise_validation"]
            edges = magnitude_edges(events_df["magnitude"].to_numpy())
        
        for m_name in ["stead", "instance"]:
            if ds_name == "silivri" and m_name != "instance":
                continue # Silivri only has instance model evaluation
                
            if ds_name in ["stead", "instance"]:
                cache_path = OUTPUT_DIR / f"cache_{dataset.__class__.__name__}_{m_name}.npz"
                if not cache_path.exists():
                    print(f"Cache not found: {cache_path}")
                    continue
                    
                scores = np.load(cache_path)
                recovar_scores = scores["recovar_scores"]
                phasenet_scores = scores["phasenet_scores"]
                
                noise_recovar = recovar_scores[noise_rows]
                noise_phasenet = phasenet_scores[noise_rows]
                
                events_df["recovar"] = recovar_scores[event_rows]
                events_df["phasenet"] = phasenet_scores[event_rows]
            else:
                pass # Silivri data already loaded properly above for 'instance'
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            ax_rec = axes[0]
            ax_phase = axes[1]
            
            cmap = plt.get_cmap("viridis")
            num_bins = len(edges) - 1
            
            for i in range(num_bins):
                left = edges[i]
                right = edges[i+1]
                bin_events = events_df[(events_df["magnitude"] >= left) & (events_df["magnitude"] < right)]
                if len(bin_events) < 50:
                    continue
                    
                color = cmap(i / num_bins)
                base_label = f"[{left:.1f}, {right:.1f})"
                
                # Proportional noise scaling
                prop = len(bin_events) / len(events_df)
                sampled_noise_size = max(1, int(len(noise_recovar) * prop))
                
                for ax, model_col, noise_all in [(ax_rec, "recovar", noise_recovar), (ax_phase, "phasenet", noise_phasenet)]:
                    y_true = np.concatenate([np.ones(len(bin_events)), np.zeros(len(noise_all))])
                    y_scores = np.concatenate([bin_events[model_col].to_numpy(), noise_all])
                    weights = np.concatenate([np.ones(len(bin_events)), np.full(len(noise_all), prop)])
                    
                    valid = ~np.isnan(y_scores)
                    if not valid.any(): continue
                    
                    prec, rec, _ = precision_recall_curve(y_true[valid], y_scores[valid], sample_weight=weights[valid])
                    pr_auc = auc(rec, prec)
                    ax.plot(rec, prec, label=f"{base_label} (AUC: {pr_auc:.3f})", color=color, alpha=0.8, linewidth=2)
                    
            for ax, title in [(ax_rec, f"RECOVAR-{m_name.upper()}"), (ax_phase, f"PhaseNet-{m_name.upper()}")]:
                ax.set_title(f"{ds_name.upper()} Dataset | {title} PR Curves")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend(fontsize=9, loc='lower left', title="Mag Bin (AUC)")
                
            fig.tight_layout()
            
            out_png = OUTPUT_DIR / f"pr_curves_by_mag_{ds_name}_model_{m_name}.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig.savefig(out_png.with_suffix('.pdf'), bbox_inches="tight")
            print(f"Saved {out_png}")
            plt.close(fig)

if __name__ == "__main__":
    main()
