ONNXConformal:conformal prediction sets/intervals for model outputs

- Fast, single-binary CLI in Rust.
- Works without binding to ORT by default (reads JSONL of logits/probs or ŷ).
- Optional `onnx` feature to run ONNX models via the `ort` crate.


Initial thoughts, it should support:

- Classification (split conformal): threshold on nonconformity 1−ptrue, optional label names, logits→softmax, max-set-size cap.
- Regression: symmetric intervals using absolute residuals.
- Smoothed quantile k = ⌈(n+1)(1−α)⌉ with safe clamps for small n.
- JSONL I/O for speed and streaming friendliness.
- Clear errors, edge cases (NaNs, missing fields, tiny calib sets).

CLI Examples (non‑ONNX)

- Build: `cargo build`

- Classification
  - Calibrate from probs/logits JSONL:
    - `cargo run -- calibrate --task class --alpha 0.1 --input calib.jsonl --output calib.json`
  - Predict sets from probs/logits JSONL:
    - `cargo run -- predict --task class --calib calib.json --input scores.jsonl --output sets.jsonl --include_probs --max_set_size 5`
  - JSONL rows:
    - Calibration: `{ "probs": [0.85,0.15], "label":"ham", "labels":["ham","phish"] }` or `{ "logits": [...], "label_index": 1 }`
    - Prediction: `{ "probs": [0.81,0.19] }` or `{ "logits": [-0.2,0.1] }`

- Regression
  - Calibrate from residuals (y_true/y_pred) JSONL:
    - `cargo run -- calibrate --task regr --alpha 0.1 --input calib_regr.jsonl --output calib_regr.json`
  - Predict intervals from y_pred JSONL:
    - `cargo run -- predict --task regr --calib calib_regr.json --input preds.jsonl --output intervals.jsonl`
  - JSONL rows:
    - Calibration: `{ "y_true": 1.0, "y_pred": 1.2 }`
    - Prediction: `{ "y_pred": -2.0 }`

ONNX/ORT support (optional)

- Build with feature `onnx` to enable running ONNX models via the `ort` crate.
  - `cargo run --features onnx -- <subcommand> ...`
  - Add `--onnx-model path.onnx` (and optionally `--onnx-input`, `--onnx-output`).
- When ONNX is used, JSONL inputs contain feature vectors `x`:
  - Classification calibration rows: `{ "x": [f32,...], "label_index": usize }` or `{ "x": [...], "label": "spam", "labels": ["ham","spam"] }`.
  - Regression calibration rows: `{ "x": [f32,...], "y_true": f64 }`.
  - Prediction rows: `{ "x": [f32,...] }`.
- For classification, the ONNX model is expected to output a 1D vector of scores per class for batch size 1. These are treated as logits and passed through softmax.

Text tokenization (optional)

- Build with features `onnx,text` to enable raw-text inputs via Hugging Face tokenizers.
  - Flags: `--tokenizer tokenizer.json [--max_len N --truncation --padding]`.
  - JSONL rows use `text`: e.g., calibration `{ "text": "phishy message", "label":"phish", "labels":["ham","phish"] }`; prediction `{ "text": "..." }`.
  - Models expecting integer token IDs (int64/int32) are supported. Input/output dtypes now include `i64/i32` in addition to `f32/bool`.

Examples

- Calibrate classification from ONNX model outputs:
  - `cargo run --features onnx -- calibrate --task class --alpha 0.1 --input calib_feats.jsonl --output calib.json --onnx-model model.onnx`
- Predict classification sets by running the ONNX model:
  - `cargo run --features onnx -- predict --task class --calib calib.json --input feats.jsonl --output sets.jsonl --onnx-model model.onnx`
- Calibrate regression from ONNX model outputs:
  - `cargo run --features onnx -- calibrate --task regr --alpha 0.1 --input calib_feats_regr.jsonl --output calib_regr.json --onnx-model model.onnx`
- Predict regression intervals by running the ONNX model:
  - `cargo run --features onnx -- predict --task regr --calib calib_regr.json --input feats.jsonl --output intervals.jsonl --onnx-model model.onnx`

Sample files

- Classification (non-ONNX):
  - Calibrate: `calib.jsonl` (at repo root)
  - Predict: `examples/class_scores.jsonl`
- Regression (non-ONNX):
  - Calibrate: `examples/regr_calib.jsonl`
  - Predict: `examples/regr_preds.jsonl`

Run with sample files

- Classification:
  - `cargo run -- calibrate --task class --alpha 0.1 --input calib.jsonl --output calib.json`
  - `cargo run -- predict --task class --calib calib.json --input examples/class_scores.jsonl --output sets.jsonl --include_probs`
- Regression:
  - `cargo run -- calibrate --task regr --alpha 0.2 --input examples/regr_calib.jsonl --output calib_regr.json`
  - `cargo run -- predict --task regr --calib calib_regr.json --input examples/regr_preds.jsonl --output intervals.jsonl`


## Types of Models Supported: 

- Model-agnostic (recommended): export predictions to JSONL.
    - Classification calibration: rows with probs or logits and a true label (label or label_index).
    - Classification prediction: rows with probs or logits.
    - Regression calibration: rows with y_true and y_pred.
    - Regression prediction: rows with y_pred.
- ONNX Runtime (optional --features onnx): single-input, single-output models.
    - Input: feature vector x: [f32; D] (we feed shape [1, D]).
    - Output (class): 1-D class scores per example (treated as logits → softmax).
    - Output (regr): scalar value (we use the first element).

Framework Examples

- Via JSONL: scikit-learn, PyTorch, TensorFlow/Keras, XGBoost, LightGBM, JAX, etc. (anything that can dump predictions/labels).
- Via ONNX: models exported from the above toolchains to ONNX that meet the single-input/single-output constraint.

Constraints & Notes

- ONNX dtypes: supports f32 and bool inputs/outputs.
- Batch size: 1 (streaming). Multi-batch not supported.
- ONNX topology: exactly one input and one output tensor.
- Classification ONNX output should be logits; if your model outputs probabilities, prefer the JSONL path to avoid re-softmaxing.
- Not supported: multilabel, multi-input/multi-output ONNX graphs, structured outputs (e.g., detection/segmentation).
