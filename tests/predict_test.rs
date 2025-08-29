use onnxconformal_rs::calibrator::CalibModel;
use onnxconformal_rs::predictor::{predict_classification, predict_regression, PredConfig};
use serde::Deserialize;
use std::io::{BufReader, BufWriter, Cursor};

#[derive(Debug, Deserialize)]
struct ClassOut {
    set_indices: Vec<usize>,
    set_size: usize,
    max_prob_index: Option<usize>,
}

/// Streaming classification prediction from probability rows.
#[test]
fn test_predict_classification_stream() {
    let calib = CalibModel {
        task: "class".into(),
        alpha: 0.1,
        global_q: 0.2,
        per_label_q: None,
        labels: None,
        n: 10,
    };
    let input = [
        serde_json::json!({"probs": [0.7, 0.2, 0.1]}),
        serde_json::json!({"probs": [0.81, 0.10, 0.09]}),
    ]
    .into_iter()
    .map(|v| v.to_string())
    .collect::<Vec<_>>()
    .join("\n");

    let reader = BufReader::new(Cursor::new(input));
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig {
            max_set_size: None,
            include_probs: false,
            max_rows: None,
            #[cfg(feature = "onnx")]
            onnx: None,
        };
        predict_classification(&calib, reader, writer, cfg).unwrap();
    }
    let out_str = String::from_utf8(out_buf).unwrap();
    let lines: Vec<&str> = out_str.lines().collect();
    assert_eq!(lines.len(), 2);
    let r1: ClassOut = serde_json::from_str(lines[0]).unwrap();
    let r2: ClassOut = serde_json::from_str(lines[1]).unwrap();

    assert_eq!(r1.set_indices, Vec::<usize>::new());
    assert_eq!(r1.set_size, 0);
    assert_eq!(r1.max_prob_index, Some(0));

    assert_eq!(r2.set_indices, vec![0]);
    assert_eq!(r2.set_size, 1);
    assert_eq!(r2.max_prob_index, Some(0));
}

#[derive(Debug, Deserialize)]
struct RegrOut {
    lower: f64,
    upper: f64,
}

/// Streaming regression interval prediction from `y_pred` rows.
#[test]
fn test_predict_regression_stream() {
    let calib = CalibModel {
        task: "regr".into(),
        alpha: 0.1,
        global_q: 0.5,
        per_label_q: None,
        labels: None,
        n: 10,
    };
    let input = [
        serde_json::json!({"y_pred": 1.0}),
        serde_json::json!({"y_pred": -2.0}),
    ]
    .into_iter()
    .map(|v| v.to_string())
    .collect::<Vec<_>>()
    .join("\n");

    let reader = BufReader::new(Cursor::new(input));
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig {
            max_set_size: None,
            include_probs: false,
            max_rows: None,
            #[cfg(feature = "onnx")]
            onnx: None,
        };
        predict_regression(&calib, reader, writer, cfg).unwrap();
    }
    let out_str = String::from_utf8(out_buf).unwrap();
    let lines: Vec<&str> = out_str.lines().collect();
    assert_eq!(lines.len(), 2);
    let r1: RegrOut = serde_json::from_str(lines[0]).unwrap();
    let r2: RegrOut = serde_json::from_str(lines[1]).unwrap();
    assert!((r1.lower - 0.5).abs() < 1e-12);
    assert!((r1.upper - 1.5).abs() < 1e-12);
    assert!((r2.lower - -2.5).abs() < 1e-12);
    assert!((r2.upper - -1.5).abs() < 1e-12);
}
