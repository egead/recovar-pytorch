import os
import sys
import tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "reproducibility"))
os.chdir(os.path.join(HERE, "..", "reproducibility"))

from continuous_data_processor import ContinuousDataPreprocessor


def _make_proc():
    cat = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    pd.DataFrame({
        "station": ["ABC"],
        "p_arrival_time": ["2025-07-01T00:00:30"],
        "s_arrival_time": ["2025-07-01T00:00:35"],
    }).to_csv(cat.name, index=False)
    p = ContinuousDataPreprocessor(
        catalog_csv=cat.name,
        output_hdf5_path="_tmp.hdf5",
        output_metadata_csv_path="_tmp.csv",
        window_length=60,
        sampling_rate=100,
    )
    os.remove(cat.name)
    return p


def test_preprocess_trace_bandpass():
    p = _make_proc()
    n = 6000
    t = np.arange(n) / 100.0
    sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t) + 3.0
    out = p._preprocess_trace(sig.astype(np.float32))
    assert out.shape == (n,)
    assert np.isfinite(out).all()
    assert abs(out.mean()) < 1e-2


def test_is_window_valid():
    p = _make_proc()
    good = np.ones(6000, dtype=np.float32)
    assert p._is_window_valid(good, good, good) is True
    nan = good.copy()
    nan[0] = np.nan
    assert p._is_window_valid(nan, good, good) is False
    zeros = np.zeros(6000, dtype=np.float32)
    assert p._is_window_valid(zeros, good, good) is False
    inf = good.copy()
    inf[0] = np.inf
    assert p._is_window_valid(inf, good, good) is False


def test_check_earthquake_in_window():
    p = _make_proc()
    from obspy import UTCDateTime
    ws = UTCDateTime("2025-07-01T00:00:00")
    label, ps, ss = p._check_earthquake_in_window("ABC", ws, 60)
    assert label == "eq"
    assert ps == int(30 * 100)
    assert ss == int(35 * 100)
    label2, ps2, ss2 = p._check_earthquake_in_window("XYZ", ws, 60)
    assert label2 == "no" and ps2 is None and ss2 is None


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
    for f in ("_tmp.hdf5", "_tmp.csv"):
        if os.path.exists(f):
            os.remove(f)
    print(f"\n{passed}/{len(fns)} passed")
