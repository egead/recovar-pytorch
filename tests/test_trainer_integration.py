import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reproducibility"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "reproducibility"))

from recovar_torch import (
    RepresentationLearningMultipleAutoencoder,
    ClassifierMultipleAutoencoder,
)
from recovar_torch.config import BATCH_SIZE
from kfold_trainer import KfoldTrainer
from kfold_dynamic_trainer import KfoldDynamicTrainer

X = np.load("../data/X_train_1280sample.npy").astype(np.float32)
Y = np.load("../data/Y_train_1280sample.npy")
B = 8


class FixtureGen:
    def __init__(self, x, y, n):
        self.x, self.y, self.n = x, y, n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = (idx * B) % (len(self.x) - B)
        return self.x[s : s + B], self.y[s : s + B]


def test_static_trainer_methods():
    t = object.__new__(KfoldTrainer)
    t.epochs = 2
    t.learning_rate = 1e-4
    t.epsilon = 1e-7
    t.beta_1 = 0.99
    t.beta_2 = 0.999
    t.device = torch.device("cpu")
    model = RepresentationLearningMultipleAutoencoder().to(t.device)
    gen = FixtureGen(X, Y, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    loss = t._train_one_epoch(model, optimizer, gen)
    assert np.isfinite(loss)
    vloss = t._validate(model, gen)
    assert np.isfinite(vloss)


def test_dynamic_trainer_methods():
    t = object.__new__(KfoldDynamicTrainer)
    t.epochs = 20
    t.final_dilation_number = 64
    t.learning_rate = 1e-4
    t.epsilon = 1e-7
    t.beta_1 = 0.99
    t.beta_2 = 0.999
    t.device = torch.device("cpu")

    sched = [int(t._get_dilation_number(e)) for e in range(t.epochs)]
    assert sched[0] == 1 and sched[-1] == 64
    assert all(b >= a for a, b in zip(sched, sched[1:]))

    model = RepresentationLearningMultipleAutoencoder().to(t.device)
    classifier = ClassifierMultipleAutoencoder(model=model).to(t.device)
    gen = FixtureGen(X, Y, 6)

    x_sel, it = t._get_dilated_batch(gen, 0, 1, classifier)
    assert x_sel.shape[0] == B and it == 1

    x_sel, it = t._get_dilated_batch(gen, 0, 3, classifier)
    assert x_sel.shape[0] == BATCH_SIZE or x_sel.shape[0] == 3 * B

    optimizer = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    loss = t._train_step(model, optimizer, X[:B])
    assert np.isfinite(loss)

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(torch.any(g != 0) for g in grads)

    vloss = t._validate(model, gen)
    assert np.isfinite(vloss)


def test_per_sample_distances():
    t = object.__new__(KfoldDynamicTrainer)
    x = torch.randn(B, 3000, 3)
    y = torch.randn(B, 3000, 3)
    d = t._per_sample_l2_distance(x, y)
    assert d.shape == (B,) and torch.all(torch.isfinite(d))
    f1 = torch.randn(B, 94, 64)
    f2 = torch.randn(B, 94, 64)
    e = t._per_sample_ensemble_distance(f1, f2)
    assert e.shape == (B,) and torch.all(torch.isfinite(e))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
