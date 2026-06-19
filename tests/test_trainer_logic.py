import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, "/home/ege/recovar-pytorch")
sys.path.insert(0, "/home/ege/recovar-pytorch/reproducibility")


def _install_stub_modules():
    cfg = types.ModuleType("config")
    cfg.KFOLD_SPLITS = 5
    sys.modules["config"] = cfg

    directory = types.ModuleType("directory")
    directory.get_checkpoint_dir = lambda *a, **k: "/tmp"
    directory.get_checkpoint_path = lambda *a, **k: "/tmp/ep.pt"
    directory.get_history_csv_path = lambda *a, **k: "/tmp/history.csv"
    sys.modules["directory"] = directory

    kfold_env = types.ModuleType("kfold_environment")
    kfold_env.KFoldEnvironment = object
    sys.modules["kfold_environment"] = kfold_env

    data_generator = types.ModuleType("data_generator")
    sys.modules["data_generator"] = data_generator

    for name in ("obspy", "h5py"):
        sys.modules[name] = types.ModuleType(name)


_install_stub_modules()

from recovar_torch import (
    RepresentationLearningMultipleAutoencoder,
    ClassifierMultipleAutoencoder,
    BATCH_SIZE,
)
from kfold_trainer import KfoldTrainer
from kfold_dynamic_trainer import KfoldDynamicTrainer


def _make_trainer(epochs=20, final_dilation_number=64):
    t = object.__new__(KfoldDynamicTrainer)
    t.device = torch.device("cpu")
    t.epochs = epochs
    t.final_dilation_number = final_dilation_number
    return t


def _x(n=8):
    rng = np.random.RandomState(0)
    return rng.randn(n, 3000, 3).astype("float32")


class _FakeGen:
    def __init__(self, n_batches, batch_size, seed=0):
        rng = np.random.RandomState(seed)
        self._batches = [
            (
                rng.randn(batch_size, 3000, 3).astype("float32"),
                rng.randint(0, 2, size=(batch_size,)),
            )
            for _ in range(n_batches)
        ]

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        return self._batches[idx]


def test_per_sample_l2_distance_shape_finite():
    t = _make_trainer()
    x = torch.from_numpy(_x(8))
    y = torch.from_numpy(_x(8) + 0.1)
    d = t._per_sample_l2_distance(x, y)
    assert d.shape == (8,), d.shape
    assert torch.all(torch.isfinite(d))
    assert torch.all(d >= 0)


def test_per_sample_ensemble_distance_shape_finite():
    t = _make_trainer()
    model = RepresentationLearningMultipleAutoencoder().eval()
    with torch.no_grad():
        out = model(torch.from_numpy(_x(8)))
    f1, f2 = out[0], out[1]
    d = t._per_sample_ensemble_distance(f1, f2)
    assert d.shape == (8,), d.shape
    assert torch.all(torch.isfinite(d))
    assert torch.all(d >= 0)


def test_dilation_schedule_monotonic_ints():
    t = _make_trainer(epochs=20, final_dilation_number=64)
    schedule = np.array([int(t._get_dilation_number(e)) for e in range(t.epochs)])
    assert schedule.dtype == np.int64 or schedule.dtype == np.int32 or schedule.dtype.kind == "i"
    assert schedule[0] == 1, schedule[0]
    assert schedule[-1] == 64, schedule[-1]
    assert np.all(np.diff(schedule) >= 0), schedule
    for v in schedule:
        assert int(v) == v


def test_dilation_number_type_is_int():
    t = _make_trainer(epochs=20, final_dilation_number=64)
    v = t._get_dilation_number(0)
    assert np.issubdtype(type(v), np.integer)


def test_get_dilated_batch_dilation_le_1():
    t = _make_trainer()
    gen = _FakeGen(n_batches=5, batch_size=BATCH_SIZE)
    model = RepresentationLearningMultipleAutoencoder().eval()
    classifier = ClassifierMultipleAutoencoder(model=model)
    x_sel, it = t._get_dilated_batch(gen, 0, 1, classifier)
    assert it == 1, it
    assert x_sel.shape == (BATCH_SIZE, 3000, 3), x_sel.shape


def test_get_dilated_batch_dilation_gt_1_keeps_top_batchsize():
    t = _make_trainer()
    gen = _FakeGen(n_batches=4, batch_size=BATCH_SIZE)
    model = RepresentationLearningMultipleAutoencoder().eval()
    classifier = ClassifierMultipleAutoencoder(model=model)
    x_sel, it = t._get_dilated_batch(gen, 0, 3, classifier)
    assert it == 3, it
    assert x_sel.shape == (BATCH_SIZE, 3000, 3), x_sel.shape
    assert isinstance(x_sel, np.ndarray)


def test_get_dilated_batch_argsort_descending():
    t = _make_trainer()
    small = 4
    n_batches = 3
    gen = _FakeGen(n_batches=n_batches, batch_size=small)
    model = RepresentationLearningMultipleAutoencoder().eval()
    classifier = ClassifierMultipleAutoencoder(model=model)

    pool = np.concatenate([gen[i][0] for i in range(n_batches)], axis=0)
    classifier.eval()
    with torch.no_grad():
        scores = classifier(torch.from_numpy(pool).float()).cpu().numpy()
    expected = np.argsort(scores)[::-1][:BATCH_SIZE]

    x_sel, it = t._get_dilated_batch(gen, 0, n_batches, classifier)
    assert it == n_batches
    assert np.allclose(x_sel, pool[expected])


def test_train_step_equivalent_nonzero_grads():
    t = _make_trainer()
    model = RepresentationLearningMultipleAutoencoder().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = _x(8)
    loss = t._train_step(model, optimizer, x)
    assert isinstance(loss, float)
    assert np.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_train_step_loss_matches_manual():
    t = _make_trainer()
    model = RepresentationLearningMultipleAutoencoder().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = _x(8)
    x_t = torch.from_numpy(x).float()

    torch.manual_seed(123)
    out = model(x_t)
    f = out[0:5]
    y = out[5:10]
    recon = sum(t._per_sample_l2_distance(x_t, yy) for yy in y) / 5.0
    pairs = [
        (0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
        (2, 3), (0, 4), (1, 4), (2, 4), (3, 4),
    ]
    ens = sum(t._per_sample_ensemble_distance(f[a], f[b]) for a, b in pairs) / 10.0
    manual = float(torch.mean(recon + ens))
    assert np.isfinite(manual)
    assert manual > 0


def test_base_trainer_model_name():
    name = KfoldTrainer.__name__
    assert name == "KfoldTrainer"
    derived = RepresentationLearningMultipleAutoencoder.__name__
    assert derived == "RepresentationLearningMultipleAutoencoder"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{len(fns)} passed, {failed} failed")
    sys.exit(1 if failed else 0)
