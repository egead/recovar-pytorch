import numpy as np
import obspy
import pytest
import torch

from recovar_torch.seisbench_model import RecovarDetector

FS = 100
DUR_S = 120
N = FS * DUR_S


def _synthetic_stream(seed=0):
    rng = np.random.default_rng(seed)
    t0 = obspy.UTCDateTime("2025-01-01T00:00:00")
    st = obspy.Stream()
    tt = np.arange(N) / FS
    burst = np.exp(-((tt - 60.0) ** 2) / (2 * 2.0**2)) * np.sin(2 * np.pi * 5 * tt)
    for comp in "ZNE":
        data = rng.standard_normal(N).astype(np.float32) * 0.1
        data += (burst * (2.0 if comp == "Z" else 1.0)).astype(np.float32)
        tr = obspy.Trace(
            data,
            header={
                "starttime": t0,
                "sampling_rate": FS,
                "network": "XX",
                "station": "TEST",
                "location": "",
                "channel": f"HH{comp}",
            },
        )
        st += tr
    return st


@pytest.fixture(scope="module")
def model():
    import os
    m = RecovarDetector().eval()
    wpath = os.path.join(os.path.dirname(__file__), "..", "models", "recovar_instance.pt")
    if os.path.exists(wpath):
        m.load_representation_state_dict(wpath, map_location="cpu")
    return m


def test_metadata():
    m = RecovarDetector()
    assert m.output_type == "point"
    assert m.in_samples == 3000
    assert m.sampling_rate == 100
    assert m.component_order == "ZNE"
    assert m.labels == ["earthquake"]
    assert m.default_args["stride"] == 100
    assert m.default_args["earthquake_threshold"] == 0.5
    assert "earthquake_threshold" in m._annotate_args


def test_forward_shape():
    m = RecovarDetector().eval()
    x = torch.randn(4, 3, 3000)
    pre = m.annotate_batch_pre(x, argdict={})
    assert pre.shape == (4, 3000, 3), pre.shape
    out = m(pre)
    assert out.shape == (4, 1), out.shape
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_annotate_runs(model):
    st = _synthetic_stream()
    ann = model.annotate(st)
    assert len(ann) >= 1
    tr = ann[0]
    assert tr.stats.channel.endswith("earthquake")
    assert abs(tr.stats.sampling_rate - 1.0) < 1e-6, tr.stats.sampling_rate
    assert np.all(np.isfinite(tr.data))
    assert tr.data.min() >= -1e-6 and tr.data.max() <= 1 + 1e-6


def test_stride_changes_rate(model):
    st = _synthetic_stream()
    ann1 = model.annotate(st, stride=100)
    ann5 = model.annotate(st, stride=500)
    assert abs(ann1[0].stats.sampling_rate - 1.0) < 1e-6
    assert abs(ann5[0].stats.sampling_rate - 0.2) < 1e-6
    assert ann1[0].stats.npts > ann5[0].stats.npts


def test_classify_returns_detections(model):
    st = _synthetic_stream()
    out = model.classify(st, earthquake_threshold=0.3)
    assert hasattr(out, "detections")
    for d in out.detections:
        assert 0.0 <= d.peak_value <= 1.0
        assert d.phase == "earthquake"
