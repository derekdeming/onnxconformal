ONNXConformal:conformal prediction sets/intervals for model outputs

- Fast, single-binary CLI in Rust.
- Works without binding to ORT by default (reads JSONL of logits/probs or ŷ).
- Optional onnx feature later to run ONNX models via the ort crate.


Initial thoughts, it should support:

- Classification (split conformal): threshold on nonconformity 1−ptrue, optional label names, logits→softmax, max-set-size cap.
- Regression: symmetric intervals using absolute residuals.
- Smoothed quantile k = ⌈(n+1)(1−α)⌉ with safe clamps for small n.
- JSONL I/O for speed and streaming friendliness.
- Clear errors, edge cases (NaNs, missing fields, tiny calib sets).