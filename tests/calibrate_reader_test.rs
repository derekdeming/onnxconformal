use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel};
use std::io::{BufReader, Cursor};

#[test]
fn test_calibrate_classification_from_reader_stdin_like() {
    let input = [
        serde_json::json!({"probs": [0.6, 0.4], "label_index": 0}).to_string(),
        serde_json::json!({"probs": [0.2, 0.8], "label_index": 1}).to_string(),
    ]
    .join("\n");
    let reader = BufReader::new(Cursor::new(input));
    let cfg = CalibConfig {
        alpha: 0.1,
        mondrian: false,
        max_rows: None,
        #[cfg(feature = "onnx")]
        onnx: None,
    };
    let model = CalibModel::fit_from_reader(reader, CalibFileKind::Classification, cfg).unwrap();
    assert_eq!(model.task, "class");
    assert_eq!(model.n, 2);
}

#[test]
fn test_calibrate_regression_from_reader_stdin_like() {
    let input = [
        serde_json::json!({"y_true": 1.0, "y_pred": 1.2}).to_string(),
        serde_json::json!({"y_true": 0.0, "y_pred": -0.1}).to_string(),
    ]
    .join("\n");
    let reader = BufReader::new(Cursor::new(input));
    let cfg = CalibConfig {
        alpha: 0.2,
        mondrian: false,
        max_rows: None,
        #[cfg(feature = "onnx")]
        onnx: None,
    };
    let model = CalibModel::fit_from_reader(reader, CalibFileKind::Regression, cfg).unwrap();

    assert_eq!(model.task, "regr");
    assert_eq!(model.n, 2);
}
