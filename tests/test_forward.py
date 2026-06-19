import torch
from recovar_torch import (
    AutoencoderBlock,
    RepresentationLearningSingleAutoencoder,
    RepresentationLearningDenoisingSingleAutoencoder,
    RepresentationLearningMultipleAutoencoder,
    RepresentationLearningMultipleAutoencoderL4,
    ClassifierAutocovariance,
    ClassifierMultipleAutoencoder,
    ClassifierAugmentedAutoencoder,
)

B, T, C = 4, 3000, 3


def _x():
    torch.manual_seed(0)
    return torch.randn(B, T, C)


def test_autoencoder_block_shapes():
    m = AutoencoderBlock().eval()
    f, y = m(_x())
    assert f.shape == (B, 94, 64), f.shape
    assert y.shape == (B, T, C), y.shape


def test_single():
    m = RepresentationLearningSingleAutoencoder().eval()
    f, y, loss = m(_x())
    assert f.shape == (B, 94, 64)
    assert y.shape == (B, T, C)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_denoising_train_vs_eval():
    m = RepresentationLearningDenoisingSingleAutoencoder()
    m.train()
    f, y, loss = m(_x())
    assert y.shape == (B, T, C) and torch.isfinite(loss)
    m.eval()
    f, y, loss = m(_x())
    assert y.shape == (B, T, C) and torch.isfinite(loss)


def test_multiple():
    m = RepresentationLearningMultipleAutoencoder().eval()
    out = m(_x())
    assert len(out) == 11
    for k in range(5):
        assert out[k].shape == (B, 94, 64)
    for k in range(5, 10):
        assert out[k].shape == (B, T, C)
    assert out[10].ndim == 0 and torch.isfinite(out[10])


def test_multiple_l4():
    m = RepresentationLearningMultipleAutoencoderL4().eval()
    out = m(_x())
    assert torch.isfinite(out[10])


def test_classifier_autocovariance():
    rep = RepresentationLearningSingleAutoencoder().eval()
    clf = ClassifierAutocovariance(model=rep).eval()
    s = clf(_x())
    assert s.shape == (B,), s.shape
    assert torch.all(s >= 0) and torch.all(s <= 1)


def test_classifier_multiple():
    rep = RepresentationLearningMultipleAutoencoder().eval()
    clf = ClassifierMultipleAutoencoder(model=rep).eval()
    s = clf(_x())
    assert s.shape == (B,)
    assert torch.all(s >= 0) and torch.all(s <= 1)


def test_classifier_augmented():
    rep = RepresentationLearningSingleAutoencoder().eval()
    clf = ClassifierAugmentedAutoencoder(model=rep).eval()
    s = clf(_x())
    assert s.shape == (B,)
    assert torch.all(s >= 0) and torch.all(s <= 1)


def test_normalize_std_unit():
    from recovar_torch import NormalizeStd
    x = torch.randn(B, T, C) * 5 + 2
    y = NormalizeStd(axis=1)(x)
    std = torch.std(y, dim=1, unbiased=False)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def test_backward_multiple():
    m = RepresentationLearningMultipleAutoencoder().train()
    out = m(_x())
    out[10].backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


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
