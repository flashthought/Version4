"""
Unit tests for Windkessel cardiovascular simulation
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.physics.windkessel import WindkesselSimulator

def test_windkessel_baseline():
    sim = WindkesselSimulator(R_baseline=1.0, C_baseline=1.2, sample_rate=125.0)
    res = sim.simulate_2element(duration=10.0, heart_rate=75.0, stroke_volume=70.0)
    
    abp = res['abp']
    # Check physiological range for normal adult (60 to 140 mmHg)
    assert np.all(abp > 40.0), f"Pressure fell below 40: min was {np.min(abp)}"
    assert np.all(abp < 160.0), f"Pressure rose above 160: max was {np.max(abp)}"
    
    # Check pulsatility (SBP > DBP by at least 20 mmHg)
    sbp = np.max(abp[-250:]) # last 2 seconds
    dbp = np.min(abp[-250:])
    assert (sbp - dbp) > 20.0, f"Pulse pressure too narrow: {sbp - dbp} mmHg"
    print(f"Baseline Test Passed: SBP={sbp:.1f}, DBP={dbp:.1f}, MAP={np.mean(abp[-250:]):.1f}")

def test_septic_shock_vasodilation():
    sim = WindkesselSimulator(R_baseline=1.0, C_baseline=1.2, sample_rate=125.0)
    
    # Vasodilation: R drops from 1.0 to 0.5 mmHg*s/mL at t=10s
    def r_shock(t):
        if t < 10.0:
            return 1.0
        elif t < 20.0:
            return 1.0 - 0.5 * ((t - 10.0) / 10.0)
        else:
            return 0.5

    res = sim.simulate_2element(duration=30.0, heart_rate=75.0, stroke_volume=70.0, R_func=r_shock)
    abp = res['abp']
    
    # MAP before shock (t in [5, 10]) vs MAP after shock (t in [25, 30])
    map_baseline = np.mean(abp[5*125 : 10*125])
    map_shock = np.mean(abp[25*125 : 30*125])
    
    assert map_shock < map_baseline - 20.0, f"Shock did not induce hypotension: base={map_baseline}, shock={map_shock}"
    print(f"Shock Vasodilation Test Passed: Base MAP={map_baseline:.1f} -> Shock MAP={map_shock:.1f}")

def test_ppg_generation():
    sim = WindkesselSimulator()
    res = sim.simulate_2element(duration=5.0)
    ppg = sim.generate_synthetic_ppg(res['abp'])
    
    assert len(ppg) == len(res['abp'])
    assert np.min(ppg) >= 0.0 and np.max(ppg) <= 1.0
    print("PPG Generation Test Passed")

if __name__ == '__main__':
    test_windkessel_baseline()
    test_septic_shock_vasodilation()
    test_ppg_generation()
    print("ALL PHYSICS TESTS PASSED!")
