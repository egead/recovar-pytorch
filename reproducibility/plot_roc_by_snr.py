import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from reproducibility.cross_dataset_snr_f1_fnr_analysis import get_dataset_obj, test_descriptors, snr_edges
from sklearn.metrics import roc_curve, auc

CACHE_DIR = REPO / "cross_dataset_mag_analysis"
OUTPUT_DIR = REPO / "cross_dataset_snr_f1_fnr_analysis"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS = ["stead", "instance"]
    
    for ds_name in DATASETS:
        dataset = get_dataset_obj(ds_name)
        descriptors = test_descriptors(dataset.metadata, ds_name)
        
        events_df = descriptors.loc[descriptors["label"].eq("eq")].copy()
        noise_rows = descriptors["label"].eq("no").to_numpy()
        event_rows = descriptors["label"].eq("eq").to_numpy()
        
        edges = snr_edges(events_df["snr"].to_numpy())
        
        for m_name in ["stead", "instance"]:
            cache_path = CACHE_DIR / f"cache_{dataset.__class__.__name__}_{m_name}.npz"
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
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            ax_rec = axes[0]
            ax_phase = axes[1]
            
            cmap = plt.get_cmap("plasma")
            num_bins = len(edges) - 1
            
            for i in range(num_bins):
                left = edges[i]
                right = edges[i+1]
                bin_events = events_df[(events_df["snr"] >= left) & (events_df["snr"] < right)]
                if len(bin_events) < 50:
                    continue
                    
                color = cmap(i / num_bins)
                base_label = f"[{left:.1f}, {right:.1f})"
                
                # For ROC, no noise scaling is needed because FPR inherently normalizes by total noise size.
                for ax, model_col, noise_all in [(ax_rec, "recovar", noise_recovar), (ax_phase, "phasenet", noise_phasenet)]:
                    y_true = np.concatenate([np.ones(len(bin_events)), np.zeros(len(noise_all))])
                    y_scores = np.concatenate([bin_events[model_col].to_numpy(), noise_all])
                    
                    valid = ~np.isnan(y_scores)
                    if not valid.any(): continue
                    
                    fpr, tpr, _ = roc_curve(y_true[valid], y_scores[valid])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{base_label} (AUC: {roc_auc:.3f})", color=color, alpha=0.8, linewidth=2)
                    
            for ax, title in [(ax_rec, f"RECOVAR-{m_name.upper()}"), (ax_phase, f"PhaseNet-{m_name.upper()}")]:
                ax.set_title(f"{ds_name.upper()} Dataset | {title} ROC Curves by SNR")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate (Recall)")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend(fontsize=9, loc='lower right', title="SNR Bin (dB) (AUC)")
                
            fig.tight_layout()
            
            out_png = OUTPUT_DIR / f"roc_curves_by_snr_{ds_name}_model_{m_name}.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig.savefig(out_png.with_suffix('.pdf'), bbox_inches="tight")
            print(f"Saved {out_png}")
            plt.close(fig)

if __name__ == "__main__":
    main()
