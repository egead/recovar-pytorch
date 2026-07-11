import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from recovar_torch import RepresentationLearningMultipleAutoencoder
from seisbench_kfold_trainer import SeisBenchKfoldTrainer
from directory import get_checkpoint_path, get_history_csv_path


def find_picovar_repo():
    candidates = [
        Path.home() / "picovar",
        Path("/mnt/second_drive/ege/picovar"),
    ]
    if "PICOVAR_DIR" in os.environ:
        candidates.insert(0, Path(os.environ["PICOVAR_DIR"]))
    for candidate in candidates:
        if (candidate / "picovar").is_dir():
            return candidate
    raise FileNotFoundError("picovar repo not found; set PICOVAR_DIR")


PICOVAR_MODELS_DIR = find_picovar_repo() / "models"

DATASETS = ["ethz", "geofon", "instance", "iquique", "neic", "scedc", "stead"]
NUM_EPOCHS = 20
SPLIT = 0

for dataset in DATASETS:
    out_path = PICOVAR_MODELS_DIR / f"recovar_{dataset}_seisbench_benchmark.pt"
    if out_path.exists():
        print(f"{dataset}: {out_path} exists, skipping")
        continue

    exp_name = f"recovar_{dataset}"
    trainer = SeisBenchKfoldTrainer(
        exp_name,
        RepresentationLearningMultipleAutoencoder,
        dataset,
        SPLIT,
        epochs=NUM_EPOCHS,
        apply_resampling=False,
    )
    trainer.train()

    history = pd.read_csv(
        get_history_csv_path(exp_name, trainer.model_name, dataset, SPLIT)
    )
    if history["val_loss"].isna().all():
        print(f"{dataset}: training produced no valid loss (empty dataset?), NOT copying a model")
        continue
    best_epoch = int(history["val_loss"].idxmin())
    checkpoint = get_checkpoint_path(
        exp_name, trainer.model_name, dataset, SPLIT, best_epoch
    )
    PICOVAR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, out_path)
    print(f"{dataset}: best epoch {best_epoch} val_loss {history['val_loss'].min():.6f} -> {out_path}")
