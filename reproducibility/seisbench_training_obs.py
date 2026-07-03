import json
from os import environ
from os.path import join

with open("settings.json", "r") as file:
    settings = json.load(file)

if "SEISBENCH_CACHE_ROOT" in settings:
    environ.setdefault("SEISBENCH_CACHE_ROOT", settings["SEISBENCH_CACHE_ROOT"])
environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import shutil
import pandas as pd
import seisbench.data as sbd

from recovar_torch import RepresentationLearningMultipleAutoencoder
from seisbench_kfold_trainer import SeisBenchKfoldTrainer
from directory import get_checkpoint_path, get_history_csv_path
from config import SAMPLING_FREQ, PHASE_ENSURING_MARGIN

TIME_WINDOW = settings["SEISBENCH_DATASETS"]["obs"]["time_window"]
PICOVAR_MODELS_DIR = "/mnt/second_drive/ege/picovar/models"
NUM_EPOCHS = 20
EXP_NAME = "recovar_obs"
DATASET = "obs"


def _c(m, name):
    low = {c.lower(): c for c in m.columns}
    if name.lower() not in low:
        raise KeyError(f"{name} not in metadata")
    return low[name.lower()]


full = sbd.OBS(component_order="Z12", sampling_rate=None)
m = full.metadata
rate = m[_c(m, "trace_sampling_rate_hz")].astype(float)
duration = m[_c(m, "trace_npts")].astype(float) / rate
p100 = m[_c(m, "trace_P_arrival_sample")].astype(float) * (SAMPLING_FREQ / rate)

long_enough = rate.notna() & (duration >= TIME_WINDOW)
event_mask = long_enough & (p100 <= (TIME_WINDOW - PHASE_ENSURING_MARGIN) * SAMPLING_FREQ)
noise_mask = long_enough & (p100 >= TIME_WINDOW * SAMPLING_FREQ)

event_dataset = full.filter(event_mask, inplace=False)
noise_dataset = full.filter(noise_mask, inplace=False)
print(f"events {len(event_dataset.metadata)}  noise {len(noise_dataset.metadata)}  of {len(m)} traces", flush=True)

for ds in (event_dataset, noise_dataset):
    md = ds.metadata
    r = md[_c(md, "trace_sampling_rate_hz")].astype(float)
    for name in ("trace_P_arrival_sample", "trace_S_arrival_sample"):
        try:
            col = _c(md, name)
        except KeyError:
            continue
        md[col] = md[col].astype(float) * (SAMPLING_FREQ / r)

trainer = SeisBenchKfoldTrainer(
    EXP_NAME, RepresentationLearningMultipleAutoencoder, DATASET, 0,
    epochs=NUM_EPOCHS, apply_resampling=False,
    event_dataset=event_dataset, noise_dataset=noise_dataset,
)
trainer.train()

hist = pd.read_csv(get_history_csv_path(EXP_NAME, trainer.model_name, DATASET, 0))
best_epoch = int(hist["val_loss"].idxmin())
src = get_checkpoint_path(EXP_NAME, trainer.model_name, DATASET, 0, best_epoch)
dst = join(PICOVAR_MODELS_DIR, "recovar_obs.pt")
shutil.copyfile(src, dst)
print(f"best epoch {best_epoch}  val_loss {hist['val_loss'].min():.4f}  -> {dst}", flush=True)
