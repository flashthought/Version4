# ISO 14971 Risk Management Log

This document outlines the risk management process for the **Physics-Informed Hemodynamic Forecaster (PIHF)** in accordance with ISO 14971 (Application of risk management to medical devices).

## Risk Matrix Definition
* **Probability ($P$)**: 1 (Improbable) to 5 (Frequent)
* **Severity ($S$)**: 1 (Negligible) to 5 (Catastrophic)
* **Risk Index ($R$)**: $P \times S$. 
  * $R \le 4$: Acceptable
  * $5 \le R \le 9$: As Low As Reasonably Practicable (ALARP)
  * $R \ge 10$: Unacceptable (Requires mandatory mitigation)

---

## Hazard Analysis & Risk Register

### Hazard 1: Sensor Disconnection / Artifact (Data Quality)
* **Description**: The patient moves or a sensor disconnects, feeding pure noise or flatlines into the predictive model.
* **Initial Risk**: $P=4, S=3 \rightarrow R=12$ (Unacceptable)
* **Mitigation**: Implemented an automated Signal Quality Index (SQI) module (`src/preprocessing/sqi.py`) that checks for amplitude variance, saturation, and valid heart rate matching before running inference.
* **Residual Risk**: $P=1, S=3 \rightarrow R=3$ (Acceptable)
* **Verification**: `test_preprocessing.py::test_sqi_abstention_flatline`

### Hazard 2: Hallucinated / Non-Physiological Output (Algorithmic Risk)
* **Description**: The Deep Learning model outputs a blood pressure trajectory that is mathematically impossible based on vascular resistance limits, misleading the clinician.
* **Initial Risk**: $P=3, S=4 \rightarrow R=12$ (Unacceptable)
* **Mitigation**: 
  1. Parameter estimation uses strict $\exp()$ projections to ensure resistance ($R$) and compliance ($C$) are strictly positive.
  2. Training objective includes a Physics-Informed Loss (`src/models/physics_loss.py`) that heavily penalizes Windkessel ODE violations.
* **Residual Risk**: $P=1, S=4 \rightarrow R=4$ (Acceptable)
* **Verification**: `test_model.py::test_parameter_positivity_constraint`, `model_comparison_results.csv` showing >99% reduction in physics residual.

### Hazard 3: Negative Resistance Prediction (Mathematical Failure)
* **Description**: The neural network outputs a negative vascular resistance, which is physically impossible and would cause the Windkessel ODE solver to diverge or flip signs.
* **Initial Risk**: $P=2, S=4 \rightarrow R=8$ (ALARP)
* **Mitigation**: Instead of predicting $R$ directly, the model predicts the latent offset $r$ and computes $R = R_0 \exp(r)$. This guarantees $R > 0$ for all real inputs.
* **Residual Risk**: $P=1, S=4 \rightarrow R=4$ (Acceptable)
* **Verification**: `test_model.py::test_parameter_positivity_constraint`

### Hazard 4: Clinician Over-reliance / Lack of Transparency (Usability Risk)
* **Description**: The clinician trusts a false-positive hypotensive forecast without understanding why the model made the prediction, leading to unnecessary vasopressor administration.
* **Initial Risk**: $P=3, S=3 \rightarrow R=9$ (ALARP)
* **Mitigation**: 
  1. The Intended Use explicitly states the device is adjunctive.
  2. Implementation of an Explainability Module (`src/evaluation/explainability.py`) using Integrated Gradients to show exactly which segments of the ECG/PPG waves drove the prediction, allowing clinicians to visually vet the model's logic.
* **Residual Risk**: $P=2, S=3 \rightarrow R=6$ (ALARP - Acceptable with monitoring)
* **Verification**: Output of `ig_attribution_map.png` demonstrating transparent attribution mapping.

---

## Conclusion
All identified risks have been mitigated to an acceptable level ($R \le 9$) through architectural constraints (physics-informed loss, strictly positive parameter mappings), data quality checks, and transparency modules. The medical benefits of early hypotensive warning outweigh the residual risks.
