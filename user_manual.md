# User Manual

`recovar_torch` is a PyTorch port of [RECOVAR](https://arxiv.org/abs/2407.18402), an unsupervised earthquake detector. It is trained on unlabeled waveform windows and outputs one earthquake score in `[0, 1)` per window.

## Input format

Each window is 30 s of 3-component data at 100 Hz.

|                     |                                                                                                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Array shape         | `(N, 3000, 3)`, float32 (windows, time samples, channels)                                                                                                                  |
| Channel order       | Must match the training data. Z, N, E in `RecovarDetector`, file order in the `reproducibility/` HDF5 pipeline.                                                            |
| Sampling rate       | 100 Hz (`recovar_torch.config.SAMPLING_FREQ`)                                                                                                                              |
| Preprocessing       | 1 to 20 Hz bandpass, then demean                                                                                                                                           |
| Model preprocessing | each channel is divided by its standard deviation inside the model                                                                                                         |

Data stored as `(N, 3, 3000)` (for example SeisBench output) must be transposed to `(N, 3000, 3)`.

The sample files in `data/` are in this format:

| File | Shape | Content |
|---|---|---|
| `data/X_train_1280sample.npy` | `(1280, 3000, 3)` | training windows |
| `data/Y_train_1280sample.npy` | `(1280,)` | labels, 1 = earthquake, 0 = noise (not used in training) |
| `data/X_test_1280sample.npy` | `(1280, 3000, 3)` | test windows |
| `data/Y_test_1280sample.npy` | `(1280,)` | labels, 1 = earthquake, 0 = noise |
| `data/test.mseed` | continuous 3-component record | input for `seisbench_demo.ipynb` |

## Models

RECOVAR has two parts. A representation model (autoencoder) is trained on unlabeled windows. A classifier wraps the trained representation model and turns its latent features into an earthquake score. 

| Representation model | Compatible classifiers |
|---|---|
| `RepresentationLearningMultipleAutoencoder` (5-autoencoder ensemble, used for all released weights) | `ClassifierMultipleAutoencoder` |
| `RepresentationLearningMultipleAutoencoderL4` | `ClassifierMultipleAutoencoder` |
| `RepresentationLearningSingleAutoencoder` | `ClassifierAutocovariance`, `ClassifierAugmentedAutoencoder` |
| `RepresentationLearningDenoisingSingleAutoencoder` | `ClassifierAutocovariance`, `ClassifierAugmentedAutoencoder` |

## Pretrained weights

All files in `models/` are `state_dict`s of `RepresentationLearningMultipleAutoencoder(input_noise_std=1e-6, eps=1e-27)`.

| File | Training dataset |
|---|---|
| `models/recovar_instance.pt` | INSTANCE |
| `models/recovar_stead.pt` | STEAD |
| `models/recovar_crlbd.pt` | CRLBD |

## Scoring windows

```python
import numpy as np
import torch
from recovar_torch import RepresentationLearningMultipleAutoencoder, ClassifierMultipleAutoencoder, BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RepresentationLearningMultipleAutoencoder(input_noise_std=1e-6, eps=1e-27).to(device)
model.load_state_dict(torch.load("models/recovar_instance.pt", map_location=device))
model.eval()
classifier = ClassifierMultipleAutoencoder(model).to(device).eval()

X = np.load("data/X_test_1280sample.npy").astype(np.float32)
scores = []
with torch.no_grad():
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[i : i + BATCH_SIZE]).to(device)
        scores.append(classifier(xb).cpu().numpy())
scores = np.concatenate(scores)
```

`scores` has shape `(N,)`. A higher score means the window is more likely to contain an earthquake. Pick a threshold (0.5 is the default in the SeisBench detector). Models also can be evaluated with ROC AUC (see: `model_test.ipynb`).

## Training

The representation model is trained by minimizing its own loss, which is the last element of the tuple returned by `forward`.

```python
out = model(xb)
loss = out[-1]
loss.backward()
```

`model_train.ipynb` contains a complete training loop with validation split, early stopping and checkpointing. See [Examples](README.md#examples) for how to run it.

## Continuous data with SeisBench

`RecovarDetector` wraps the ensemble model as a SeisBench `WaveformModel`. It resamples to 100 Hz, orders components as ZNE and applies the 1 to 20 Hz bandpass. It takes a raw ObsPy `Stream` as input.

```python
import obspy
from recovar_torch.seisbench_model import RecovarDetector

model = RecovarDetector().eval()
model.load_representation_state_dict("models/recovar_instance.pt", map_location="cpu")

st = obspy.read("data/test.mseed")
st.merge(method=1, fill_value=0)

curve = model.annotate(st, stride=100)[0]
detections = model.classify(st, stride=100, earthquake_threshold=0.5).detections
```

| Argument | Meaning |
|---|---|
| `stride` | step between consecutive 30 s windows, in samples at 100 Hz (100 = 1 s). The output curve is sampled at `100 / stride` Hz. |
| `earthquake_threshold` | score above which `classify` reports a detection |

`annotate` returns the score curve as an ObsPy trace. `classify` thresholds that curve and returns detections with start and end times.

Example: `seisbench_demo.ipynb` runs this on `data/test.mseed` and compares against PhaseNet and EQTransformer.

## Reproducibility scripts

The scripts in `reproducibility/` train and evaluate on full datasets with a fixed train/validation/test split. They read `reproducibility/settings.json`. It must be run from that directory as:

```bash
cd reproducibility
python training.py
python testing.py
```

Set these entries in `settings.json` before running:

| Key                              | Meaning                                                 |
| -------------------------------- | ------------------------------------------------------- |
| `DATASET_DIRECTORIES`            | HDF5 waveform and CSV metadata paths                    |
| `CUSTOM_DATASETS`                | custom HDF5/CSV datasets                                |
| `SEISBENCH_DATASETS`             | SeisBench datasets used by the `seisbench_*.py` scripts |
| `SEISBENCH_CACHE_ROOT`           | SeisBench download cache                                |
| `PREPROCESSED_DATASET_DIRECTORY` | preprocessed windows path                               |
| `TRAINED_MODELS_DIR`             | checkpoints path                                        |
| `RESULTS_DIR`                    | evaluation results path                                 |

| Script                                          | Purpose                                                                                                                                                                                                               |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training.py`, `testing.py`                     | train and evaluate on HDF5/CSV datasets (STEAD, INSTANCE, custom)                                                                                                                                                     |
| `seisbench_training.py`, `seisbench_testing.py` | train and evaluate on SeisBench datasets                                                                                                                                                                              |
| `seisbench_training_benchmark.py`               | train one model per SeisBench benchmark dataset                                                                                                                                                                       |
| `dynamic_training.py`, `dynamic_testing.py`     | train with score-based batch selection: each step scores a pool of batches with the current classifier and trains on the 256 highest-scoring windows. The pool grows from 1 to 64 log-spaced batches over the epochs. |
| `continuous_preprocess_script.py`               | cut continuous MSEED station data into fixed-length windows, label each window from an event catalog. Write HDF5 waveforms and a CSV metadata file                                                                    |

## HDF5 and SeisBench pipelines

The scripts in `reproducibility/` use one of two data pipelines. They are different only in how traces are read and labeled.

| | HDF5 pipeline | SeisBench pipeline |
|---|---|---|
| Scripts | `training.py`, `testing.py`, `dynamic_*.py` | `seisbench_*.py` |
| Code | `kfold_environment.py`, `data_generator.py` | `seisbench_kfold_environment.py`, `seisbench_data_generator.py` |
| Input | STEAD HDF5 and CSV, INSTANCE event and noise HDF5 and CSV, or a `CUSTOM_DATASETS` entry `{"waveforms": <hdf5>, "metadata": <csv>}` in STEAD format | event and noise datasets from `SEISBENCH_DATASETS`: a named SeisBench dataset (`seisbench_name`, split into events and noise by `trace_category`, `filter_split` keeps only the listed splits) or local `event_path` and `noise_path` |
| Labels | STEAD and custom: `trace_category` (`earthquake_local` or `noise`). INSTANCE: event file or noise file. | event dataset is earthquake, noise dataset is noise. Event traces with no pick, or with the first pick past the trace length, are evaluated as noise. |
| Trace length |  60 s (STEAD, custom) or 120 s (INSTANCE) | `time_window` in `settings.json`, else the median trace length clipped to 30 to 180 s. Shorter traces are dropped. |
| Sampling rate | must already be 100 Hz | resampled to 100 Hz by SeisBench |
| Channel order | as stored in the file | `component_order` in `settings.json` |

**1. Chunk and split**
- In: one metadata row per trace with its label and P and S pick samples.
- Does: splits traces into 5 chunks (`DATASET_CHUNKS`) by `source_id`, so each event stays in one chunk. 3 chunks are training, 1 validation, 1 test (`TRAIN_VALIDATION_SPLIT` = 0.75).
- Out: training, validation and test chunks.

**2. Crop**
- In: trace metadata.
- Does: draws a seeded random start for the 30 s window inside each trace. Earthquake windows must end at least `PHASE_ENSURING_MARGIN` (6 s) after the first pick and start at least 6 s before the last pick.
- Out: one fixed crop offset per trace.

**3. Render**
- In: raw traces and crop offsets.
- Does: shifts each trace so the crop start is at sample 0 (phase shift in the frequency domain), zeroes frequencies outside 1 to 20 Hz, keeps 3000 samples, demeans, and divides each channel by its L2 norm.
- Out: `PREPROCESSED_DATASET_DIRECTORY/<dataset>/<dataset>_subsampled_<percent>percent.hdf5` with batches of `(256, 3000, 3)` windows and labels. Created on first run and reused after that. Delete it if you change chunk, crop or trace length settings.

**4. Train** (`kfold_trainer.py`)
- In: training and validation batches.
- Does: runs Adam (learning rate 1e-4) on the model loss for a fixed number of epochs, without early stopping. Labels are not used.
- Out: `ep<epoch>.pt` for every epoch and `history.csv` with training and validation loss, in `TRAINED_MODELS_DIR/<exp_name>/<model name>/<dataset>/split0/`.

**5. Test** (`evaluator.py`, `kfold_tester.py`)
- In: the checkpoint with the lowest validation loss, and the test chunk of the test dataset (one fifth of it). Test dataset can be different from training dataset.
- Does: scores every test window, keeps earthquake windows whose P pick is at least 6 s from both window edges (`CropOffsetFilter`), and computes the ROC curve.
- Out: `scores<epoch>.csv` and `meta.csv` in `RESULTS_DIR/<exp_name>/<model name>/<classifier name>/training_<train>/testing_<test>/split0/`. `testing.py` also appends the ROC AUC to a CSV and saves the ROC plot.

### Full k-fold training

The scripts only run split 0. `settings.json` defines 5 splits (`KFOLD_SPLITS`), each with a different test chunk.

To train all splits, change `range(1)` to `range(5)` in the training script. To test them, run the testing script with `SPLIT` set to 0 to 4.

Each split saves to its own `split<k>/` folder.
