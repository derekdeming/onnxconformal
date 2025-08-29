use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel};
use onnxconformal_rs::predictor::{predict_classification, predict_regression, PredConfig};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Cursor, Read};
use std::path::PathBuf;

fn root_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

#[test]
fn example_classification_end_to_end() {
    // Calibrate from the provided calibration file at repo root
    let calib_path = root_path("calib.jsonl");
    let cfg = CalibConfig { alpha: 0.1, mondrian: false, max_rows: None };
    let model = CalibModel::fit_from_file(calib_path.to_str().unwrap(), CalibFileKind::Classification, cfg).unwrap();
    assert_eq!(model.task, "class");
    assert!(model.global_q.is_finite());
    assert!(model.n >= 1);

    // Predict from examples/class_scores.jsonl
    let scores_path = root_path("examples/class_scores.jsonl");
    let reader = BufReader::new(File::open(scores_path).unwrap());
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let pred_cfg = PredConfig { max_set_size: Some(1), include_probs: true, max_rows: None };
        predict_classification(&model, reader, writer, pred_cfg).unwrap();
    }
    let out = String::from_utf8(out_buf).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2);

    #[derive(Deserialize)]
    struct Out { set_indices: Vec<usize>, set_size: usize, max_prob_index: Option<usize> }
    let o1: Out = serde_json::from_str(lines[0]).unwrap();
    let o2: Out = serde_json::from_str(lines[1]).unwrap();
    assert!(o1.set_size <= 1);
    assert!(o2.set_size <= 1);
    assert_eq!(o1.max_prob_index, Some(0)); // first row: probs [0.90, 0.10]
}

#[test]
fn example_regression_end_to_end() {
    // Calibrate from examples/regr_calib.jsonl
    let calib_path = root_path("examples/regr_calib.jsonl");
    let cfg = CalibConfig { alpha: 0.2, mondrian: false, max_rows: None };
    let model = CalibModel::fit_from_file(calib_path.to_str().unwrap(), CalibFileKind::Regression, cfg).unwrap();
    assert_eq!(model.task, "regr");
    assert!(model.global_q.is_finite());

    // Predict intervals for examples/regr_preds.jsonl
    let preds_path = root_path("examples/regr_preds.jsonl");
    let reader = BufReader::new(File::open(preds_path).unwrap());
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig { max_set_size: None, include_probs: false, max_rows: None };
        predict_regression(&model, reader, writer, cfg).unwrap();
    }
    let out = String::from_utf8(out_buf).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2);

    #[derive(Deserialize)]
    struct R { y_pred: f64, lower: f64, upper: f64, width: f64 }
    let r1: R = serde_json::from_str(lines[0]).unwrap();
    let r2: R = serde_json::from_str(lines[1]).unwrap();
    // width should be 2*global_q
    let w = 2.0 * model.global_q;
    assert!((r1.width - w).abs() < 1e-9);
    assert!((r2.width - w).abs() < 1e-9);
}

