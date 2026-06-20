# recovar-pytorch

PyTorch port of [RECOVAR](https://arxiv.org/abs/2407.18402)

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

```bash
python -m pytest tests/          # or: python tests/test_forward.py
jupyter notebook model_train.ipynb
```

Training/eval scripts are under`reproducibility/` and read `reproducibility/settings.json` (run from inside that dir). 
Point its `DATASET_DIRECTORIES` / `CUSTOM_DATASETS` at your data.
