import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "reproducibility"))
os.chdir(os.path.join(HERE, "..", "reproducibility"))

from recovar_torch import (
    RepresentationLearningMultipleAutoencoder,
    ClassifierMultipleAutoencoder,
)
from kfold_tester import KFoldTester
from sklearn.metrics import roc_curve, roc_auc_score

X = np.load("../data/X_test_1280sample.npy").astype(np.float32)
Y = np.load("../data/Y_test_1280sample.npy")
B = 8
NB = 4


class FakePredictGen:
    def __len__(self):
        return NB

    def __getitem__(self, i):
        s = (i * B) % (len(X) - B)
        return X[s : s + B]


def test_tester_predict_path():
    t = object.__new__(KFoldTester)
    t.device = torch.device("cpu")
    rep = RepresentationLearningMultipleAutoencoder().eval()
    clf = ClassifierMultipleAutoencoder(rep).eval()
    scores = t._predict(clf, FakePredictGen())
    assert scores.shape == (NB * B,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_checkpoint_roundtrip():
    rep = RepresentationLearningMultipleAutoencoder()
    path = "_tmp_ckpt.pt"
    torch.save(rep.state_dict(), path)
    rep2 = RepresentationLearningMultipleAutoencoder()
    rep2.load_state_dict(torch.load(path, map_location="cpu"))
    os.remove(path)
    for (k1, v1), (k2, v2) in zip(rep.state_dict().items(), rep2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_evaluator_roc_from_scores():
    import sys as _sys
    import types
    if "kfold_tester" in _sys.modules:
        pass
    import evaluator

    n = 256
    rng = np.random.RandomState(0)
    scores = rng.rand(n)
    labels = (rng.rand(n) > 0.5)
    df_score = {"eq_probabilities": scores}
    import pandas as pd
    df_score = pd.DataFrame(df_score)
    df_meta = pd.DataFrame({"label": np.where(labels, "eq", "no")})

    ev = object.__new__(evaluator.Evaluator)
    tpr, fpr, thr = ev._get_roc_vector(df_score, df_meta)
    assert len(tpr) == len(fpr) == len(thr)
    auc = roc_auc_score(labels, scores)
    assert 0.0 <= auc <= 1.0


def test_evaluator_classes_exist():
    import evaluator
    for c in ("Evaluator", "SNRFilter", "CropOffsetFilter", "TracesFilter"):
        assert isinstance(getattr(evaluator, c), type)
    assert issubclass(evaluator.SNRFilter, evaluator.TracesFilter)
    assert issubclass(evaluator.CropOffsetFilter, evaluator.TracesFilter)


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
