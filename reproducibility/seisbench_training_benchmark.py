import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from recovar_torch import RepresentationLearningMultipleAutoencoder
from seisbench_kfold_trainer import SeisBenchKfoldTrainer
from directory import get_checkpoint_path, get_history_csv_path

PICOVAR_MODELS_DIR = Path.home() / "picovar" / "models"

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
    best_epoch = int(history["val_loss"].idxmin())
    checkpoint = get_checkpoint_path(
        exp_name, trainer.model_name, dataset, SPLIT, best_epoch
    )
    PICOVAR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, out_path)
    print(f"{dataset}: best epoch {best_epoch} val_loss {history['val_loss'].min():.6f} -> {out_path}")
