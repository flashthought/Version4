# Deployment Inference Layer

The app should call the inference layer, not the training scripts.

## Deployment Artifacts

After the full run, the app-importable model files are:

```text
data/models/deployment/physics_informed_hemo.pt
data/models/deployment/physics_informed_hemo.onnx
data/models/model_metadata.json
```

Use `physics_informed_hemo.pt` for a Python/PyTorch or LibTorch app. Use
`physics_informed_hemo.onnx` for a non-Python app that can run ONNX Runtime.

The stable tensor interface is:

```text
inputs:
  ecg           float32 [batch, 1, 1250]
  ppg           float32 [batch, 1, 1250]
  abp           float32 [batch, 1, 1250]
  intervention  float32 [batch, 1, 1250]

outputs:
  forecast_abp  float32 [batch, 15]
  R             float32 [batch, 1]
  C             float32 [batch, 1]
```

If a channel is missing at runtime, pass zeros for that channel and preserve the
quality/warning behavior in `src/inference`.

## Runtime Flow

```text
patient monitor stream
  -> StreamingWindowBuffer
  -> MonitorWindow
  -> HemodynamicPredictor.predict(...)
  -> PredictionResult JSON/dict
```

## Minimal Example

```python
from src.inference import HemodynamicPredictor
from src.inference.streaming_buffer import StreamingWindowBuffer

predictor = HemodynamicPredictor(
    model_path="data/models/physics-informed_real_finetuned.pth",
    metadata_path="data/models/model_metadata.json",
)

buffer = StreamingWindowBuffer(fs=125.0, window_sec=10.0, patient_id="demo")

for sample in monitor_stream:
    buffer.append(
        ecg=sample.get("ecg"),
        ppg=sample["ppg"],
        abp=sample.get("abp"),
        intervention=sample.get("intervention"),
    )
    if buffer.ready:
        result = predictor.predict(buffer.to_window(timestamp=sample["timestamp"]))
        send_to_app(result.to_dict())
```

## Output States

- `ok`: valid prediction.
- `low_confidence`: prediction returned, but warnings should be displayed.
- `abstained`: no prediction because signal quality is unsafe.
- `error`: inference failed.

The app should never hide warnings. If `status` is not `ok`, display the
reason to the user and avoid clinical interpretation.
