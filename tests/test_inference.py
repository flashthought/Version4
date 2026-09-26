"""Tests for deployment-oriented inference utilities."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.preprocessing import MonitorPreprocessor
from src.inference.quality_gate import assess_runtime_quality
from src.inference.schemas import MonitorWindow
from src.inference.streaming_buffer import StreamingWindowBuffer


def test_monitor_preprocessor_shapes():
    fs = 125.0
    t = np.arange(0, 10, 1 / fs)
    window = MonitorWindow(
        fs=fs,
        ecg=np.sin(2 * np.pi * 1.2 * t),
        ppg=np.sin(2 * np.pi * 1.2 * t + 0.4),
        abp=80 + 20 * np.sin(2 * np.pi * 1.2 * t),
        intervention=np.zeros_like(t),
    )
    prepared = MonitorPreprocessor(target_fs=125.0, window_sec=10.0).prepare(window)
    assert prepared.ecg.shape == (1250,)
    assert prepared.ppg.shape == (1250,)
    assert prepared.abp.shape == (1250,)


def test_quality_gate_abstains_on_bad_ppg():
    fs = 125.0
    x = np.ones(1250)
    report = assess_runtime_quality(
        ecg=x,
        ppg=x,
        abp=80 + np.zeros_like(x),
        intervention=np.zeros_like(x),
        fs=fs,
    )
    assert report.status == "invalid"
    assert not report.is_usable


def test_streaming_buffer_window_creation():
    fs = 125.0
    buf = StreamingWindowBuffer(fs=fs, window_sec=10.0, patient_id="demo")
    for i in range(1250):
        buf.append(ppg=float(i % 10), ecg=0.1, abp=80.0, intervention=0.0)
    assert buf.ready
    window = buf.to_window(timestamp="2026-09-25T00:00:00Z")
    assert window.patient_id == "demo"
    assert len(window.ppg) == 1250
