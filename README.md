# recovar-pytorch

PyTorch port of [RECOVAR](https://arxiv.org/abs/2407.18402)

See the [User Manual](user_manual.md) for the input format, models, pretrained weights, the SeisBench detector and the reproducibility scripts.

## Setup

Requires **Python 3.10+**. Install `torch` first (GPU or CPU)

```bash
conda create -n recovar-pytorch python=3.10
conda activate recovar-pytorch
```

### GPU (CUDA)

Using the **cu118** wheels 
```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

Verify the GPU is seen:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Device is auto-detected (`torch.cuda.is_available()`) 

## Run

Training/eval scripts are under`reproducibility/` and read `reproducibility/settings.json` (run from inside that dir). 
Point its `DATASET_DIRECTORIES` / `CUSTOM_DATASETS` at your data.

## Examples

Run the notebooks from the repository root.

### Training: `model_train.ipynb`

Trains `RepresentationLearningMultipleAutoencoder` on `data/X_train_1280sample.npy`.

```bash
jupyter notebook model_train.ipynb
```

- Runs 2 epochs by default. Set `RECOVAR_EPOCHS` to change it, e.g. `RECOVAR_EPOCHS=20 jupyter notebook model_train.ipynb`.
- Holds out the last 10% of windows for validation and stops early after 2 epochs without validation improvement.
- Saves a checkpoint per epoch to `checkpoints/autoencoder_epoch_XX.pt` and the best validation weights to `checkpoints/representation_cross_covariances.pt`.

### Testing: `model_test.ipynb`

Scores `data/X_test_1280sample.npy` and plots the ROC curve with AUC against `data/Y_test_1280sample.npy`.

```bash
jupyter notebook model_test.ipynb
```

- Uses `models/recovar_instance.pt` by default. To test the model trained in `model_train.ipynb`, set `MODEL_PATH = "checkpoints/representation_cross_covariances.pt"`.
- To try another representation model, change the model cell and the classifier cell together. Compatible pairs are listed in the [User Manual](user_manual.md#models).

### Testing on CRLBD: `crlbd_test.ipynb`

Scores the CRLBD dataset with `models/recovar_instance.pt` and plots the ROC curve with AUC.

```bash
jupyter notebook crlbd_test.ipynb
```

- Needs the CRLBD event and noise datasets in SeisBench format. Set `EVENT_DATA_PATH` and `NOISE_DATA_PATH` to their directories.
- Event windows are 30 s cut 5 s before the P pick (S pick if there is no P). Noise windows start at the trace start. 
- Windows are bandpassed 1 to 20 Hz before scoring.
- Set `MAX_TRACES` to an integer for a quick run on a subset.
