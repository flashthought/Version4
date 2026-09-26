"""
Unit tests for signal preprocessing and quality assessment
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.physics.windkessel import WindkesselSimulator
from src.preprocessing.filters import filter_ecg, filter_ppg, normalize_signal
from src.preprocessing.fiducials import detect_ecg_rpeaks, detect_ppg_fiducials, compute_pat, compute_cardiovascular_features
from src.preprocessing.sqi import evaluate_window_quality, check_flatline_or_clipping

def test_filtering_and_fiducials():
    sim = WindkesselSimulator(sample_rate=125.0)
    res = sim.simulate_2element(duration=10.0, heart_rate=75.0) # 10s at 75 bpm -> ~12.5 beats
    
    # Generate synthetic ECG triggers & PPG
    t = res['t']
    t_cycle = 60.0 / 75.0
    
    # Synthetic clean ECG
    ecg = np.zeros_like(t)
    for k in range(int(10.0 / t_cycle)):
        t_peak = k * t_cycle + 0.05
        idx = int(t_peak * 125.0)
        if idx < len(ecg):
            ecg[idx] = 1.0 # impulse
            if idx > 0: ecg[idx-1] = 0.5
            if idx < len(ecg)-1: ecg[idx+1] = 0.5
            
    ppg = sim.generate_synthetic_ppg(res['abp'], transit_time_ms=180.0)
    
    # Add noise
    ecg_noisy = sim.add_noise(ecg, snr_db=25.0)
    ppg_noisy = sim.add_noise(ppg, snr_db=25.0)
    
    # Filter
    ecg_clean = filter_ecg(ecg_noisy, fs=125.0)
    ppg_clean = filter_ppg(ppg_noisy, fs=125.0)
    
    # Fiducials
    feats = compute_cardiovascular_features(ecg_clean, ppg_clean, fs=125.0)
    print(f"Features extracted: HR={feats['heart_rate_bpm']:.1f} bpm, Matched PATs={feats['num_matched_pats']}, Mean PAT={feats['mean_pat_ms']:.1f} ms")
    
    assert 70.0 <= feats['heart_rate_bpm'] <= 80.0, f"HR estimation off: {feats['heart_rate_bpm']}"
    assert feats['num_matched_pats'] >= 10, f"Too few PAT matches: {feats['num_matched_pats']}"
    assert 100.0 <= feats['mean_pat_ms'] <= 250.0, f"PAT out of bounds: {feats['mean_pat_ms']}"
    print("Filtering and Fiducials Test Passed")

def test_sqi_abstention():
    sim = WindkesselSimulator(sample_rate=125.0)
    res = sim.simulate_2element(duration=10.0, heart_rate=75.0)
    ppg = sim.generate_synthetic_ppg(res['abp'])
    ecg = np.sin(2 * np.pi * 1.25 * res['t']) # simple surrogate
    
    # Valid window check
    q_valid = evaluate_window_quality(ecg, ppg, num_ecg_rpeaks=12, num_ppg_pulses=12, fs=125.0)
    assert q_valid['is_valid'] == True, f"Valid signal rejected: {q_valid}"
    print("SQI Valid Window Test Passed")
    
    # Corrupted flatline check
    corrupted_ppg = ppg.copy()
    corrupted_ppg[200:400] = 0.0 # flatline for 200 samples
    q_corrupt = evaluate_window_quality(ecg, corrupted_ppg, num_ecg_rpeaks=12, num_ppg_pulses=12, fs=125.0)
    assert q_corrupt['is_valid'] == False, f"Flatline was not rejected: {q_corrupt}"
    assert q_corrupt['ppg_flatline'] == True
    print("SQI Flatline Abstention Test Passed")

if __name__ == '__main__':
    test_filtering_and_fiducials()
    test_sqi_abstention()
    print("ALL PREPROCESSING TESTS PASSED!")
