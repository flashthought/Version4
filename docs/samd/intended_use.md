# Software as a Medical Device (SaMD) - Intended Use Statement

## 1. Device Description
The **Physics-Informed Hemodynamic Forecaster (PIHF)** is a Software as a Medical Device (SaMD) that utilizes a deep learning architecture constrained by physiological differential equations (the Windkessel model). The PIHF system processes continuous, non-invasive waveform data—specifically Electrocardiogram (ECG) and Photoplethysmogram (PPG)—to estimate unobserved latent cardiovascular parameters (Systemic Vascular Resistance and Arterial Compliance) and forecast future Arterial Blood Pressure (ABP) trajectories up to 15 seconds in advance.

## 2. Intended Use / Intended Purpose
The PIHF is intended to be used by healthcare professionals (e.g., intensivists, critical care nurses, and anesthesiologists) in high-acuity clinical settings (Intensive Care Units and Operating Rooms) as an adjunctive decision-support tool. It provides real-time, non-invasive continuous estimation and short-term forecasting of arterial blood pressure and vascular resistance. 

The device is intended to provide an early warning of acute hemodynamic instability (e.g., sudden hypotensive episodes associated with septic shock) to allow for proactive clinical intervention, such as the administration of vasopressors or fluid resuscitation. 

## 3. Indications for Use
The PIHF is indicated for use:
* In adult patients (≥18 years of age).
* Who require continuous hemodynamic monitoring in a critical care or perioperative setting.
* When continuous invasive arterial line monitoring is either unavailable, contraindicated, or awaiting placement.

## 4. Contraindications
* The device is not intended to replace invasive arterial blood pressure monitoring when clinically indicated.
* The device is not indicated for pediatric or neonatal populations.
* The device is contraindicated for patients with severe cardiac arrhythmias (e.g., uncontrolled atrial fibrillation) that prevent reliable feature extraction (e.g., Pulse Arrival Time) from ECG/PPG waveforms.
* The device should not be used as the sole basis for clinical diagnosis or therapeutic intervention.

## 5. Clinical Benefit
By providing continuous, non-invasive estimates of vascular parameters and forecasting impending hypotensive events, the PIHF aims to reduce the incidence and duration of severe intraoperative and ICU hypotension, thereby potentially reducing the risk of end-organ ischemia, acute kidney injury, and myocardial injury. The physics-informed constraint ensures that model outputs obey cardiovascular laws, significantly reducing the risk of mathematically "hallucinated" physiological states common in data-only models.
