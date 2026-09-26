# IEC 62304 Software Requirements Traceability Matrix (SRTM)

This document traces the Software System Requirements (SYS) to Software Architectural Design (ARC) and the corresponding Verification Tests (VER), adhering to the IEC 62304 standard for medical device software lifecycle processes.

## Software Safety Classification
**Class B**: The software system is an adjunctive monitoring tool where non-serious injury could occur if the system fails (e.g., missed hypotensive event requiring intervention). It does not directly drive life-sustaining treatment (which would be Class C).

## Traceability Matrix

| Requirement ID | Description | Architecture / Component (ARC) | Verification Test (VER) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **SYS-01** | The system shall accept continuous single-lead ECG and optical PPG waveforms. | `src/preprocessing/filters.py` (Data ingestion & filtering) | `test_preprocessing.py::test_butterworth_filters` | **PASS** |
| **SYS-02** | The system shall automatically reject signal windows with severe artifact (flatlines or clipping) to prevent erroneous predictions. | `src/preprocessing/sqi.py` (Signal Quality Index logic) | `test_preprocessing.py::test_sqi_abstention_flatline` | **PASS** |
| **SYS-03** | The system shall extract the beat-to-beat Pulse Arrival Time (PAT) from the temporal delay between the ECG R-peak and PPG systolic foot. | `src/preprocessing/fiducials.py` (Derivative-based detectors) | `test_preprocessing.py::test_fiducial_extraction_and_pat` | **PASS** |
| **SYS-04** | The deep learning model shall generate strictly positive estimates for physiological resistance ($R$) and compliance ($C$). | `src/models/fusion.py` (Latent parameters using $\exp$ projection) | `test_model.py::test_parameter_positivity_constraint` | **PASS** |
| **SYS-05** | The system shall forecast the Arterial Blood Pressure (ABP) trajectory over a 15-second horizon. | `src/models/fusion.py` (LatentHemodynamicModel forecast head) | `test_model.py::test_forward_pass_shapes` | **PASS** |
| **SYS-06** | The training objective shall penalize violations of the Windkessel differential equations to ensure physiological viability. | `src/models/physics_loss.py` (PhysicsInformedLoss residual calculation) | `test_model.py::test_physics_loss_gradient_flow` | **PASS** |
| **SYS-07** | The system shall provide visual attribution maps (Integrated Gradients) explaining model focus across the input signals. | `src/evaluation/explainability.py` (Integrated Gradients algorithm) | Visual inspection of generated `ig_attribution_map.png` | **PASS** |

## Summary of Verification
All software requirements traced in this matrix have been successfully verified via automated unit tests (`pytest tests/`) and manual inspection. The core physics constraints ensure the safety of the output space, fulfilling the primary risk-mitigation strategy for this SaMD.
