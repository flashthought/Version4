# Stage-2 BME Design Project Report: Physics-Informed Latent Hemodynamic State Estimation

## 1. Introduction
* **Clinical Problem**: Hypotension in septic shock and peri-operative settings leads to organ ischemia. Continuous invasive arterial lines are not always available.
* **Proposed Solution**: A physics-informed deep learning framework to estimate latent cardiovascular parameters (Resistance and Compliance) and forecast Arterial Blood Pressure (ABP) using non-invasive ECG and PPG waveforms.
* **Scope**: Evaluated on synthetic cardiovascular models and the public MIMIC-BP dataset to establish observability and identifiability.

## 2. Methodology
### 2.1 Public Dataset Characterisation (Phase 1)
* **Data Sources**: MIMIC-BP dataset containing highly sampled ECG, PPG, and ABP waveforms.
* **Signal Processing Pipeline**: Zero-phase Butterworth filtering, Pulse Arrival Time (PAT) fiducial extraction, and an automated Signal Quality Index (SQI) for rejecting clinical artifact (flatlines/clipping).

### 2.2 Reduced-Order Cardiovascular Simulator (Phase 2)
* **Windkessel Model**: A 3-element ODE model used to generate synthetic trajectories for Resistance ($R$), Compliance ($C$), and cardiac inflow.
* **Simulation Benefit**: Provided the controlled ground-truth required for parameter recovery metrics, which are absent in real clinical datasets.

### 2.3 Deep Learning Architecture & Physics Loss (Phase 3 & 4)
* **Architecture**: Dual 1D-Dilated Convolutional Encoders with Cross-Modal Attention.
* **Positivity Constraint**: Utilizing a log-parameterization ($R = R_0 \exp(r)$) to guarantee physically possible cardiovascular states.
* **Physics-Informed Loss**: Penalizing violations of the Windkessel ODE directly in the loss function to drastically reduce mathematical "hallucinations".

## 3. Results
### 3.1 Identifiability Experiments (Synthetic Data)
* **Observation Regimes**: Comparing recovery under PPG alone vs. PPG+ECG vs. full dynamic observation.
* **Outcome**: The addition of dynamic intervention (Regime 4) significantly reduced parameter uncertainty compared to static observation.

### 3.2 Forecasting Performance (Synthetic & Real Data)
* **Synthetic Cohort**: The Physics-Informed model achieved comparable MAE to the data-only baseline while reducing the physics residual by >99%.
* **MIMIC-BP Real-World Evaluation**: *(Insert MAE/RMSE results from evaluate_mimic.py here)*

### 3.3 Explainability
* **Integrated Gradients**: Demonstrated the network's interpretability by mapping attribution scores directly back to the ECG QRS complex and PPG systolic foot.

## 4. Limitations
* **Dataset Bias**: MIMIC-BP and VitalDB data segments are relatively short, limiting longitudinal analysis over multiday ICU stays.
* **Ground Truth Absence**: Real public waveform datasets lack continuous ground truth for systemic vascular resistance and arterial compliance, preventing direct parameter recovery validation on human patients.
* **Simulation Gap**: Synthetic cardiovascular models inherently simplify the complexity and heterogeneity of real human physiology.

## 5. Requirements for Future Septic-Shock Validation
* Translation to clinical care requires validation on a dedicated septic-shock cohort with continuous invasive thermodilution (e.g., Swan-Ganz) or calibrated PiCCO monitoring to provide real-world ground truth for vascular resistance.

## 6. Regulatory Documentation (SaMD)
* *(Refer to the Intended Use, IEC 62304 Traceability, and ISO 14971 Risk Log provided in the `docs/samd` directory).*
