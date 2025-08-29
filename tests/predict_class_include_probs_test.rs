use onnxconformal_rs::calibrator::CalibModel;
use onnxconformal_rs::predictor::{predict_classification, PredConfig};
use serde::Deserialize;
use std::io::{BufReader, BufWriter, Cursor};

#[derive(Deserialize)]
struct ClassOutFull {
    set_indices: Vec<usize>,
    set_labels: Option<Vec<String>>,
    set_size: usize,
    max_prob_label: Option<String>,
    max_prob_index: Option<usize>,
    set_probs: Option<Vec<f64>>,
}

/// Classification sets: includes labels and probs, with and without top‑k truncation.
#[test]
fn test_predict_classification_labels_and_include_probs_truncation() {
    let calib = CalibModel {
        task: "class".into(),
        alpha: 0.1,
        global_q: 0.7,
        per_label_q: None,
        labels: Some(vec!["a".into(), "b".into(), "c".into()]),
        n: 42,
    };

    let input = serde_json::json!({"probs": [0.6, 0.3, 0.1]}).to_string();
    let reader = BufReader::new(Cursor::new(input));
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig { max_set_size: None, include_probs: true, max_rows: None, #[cfg(feature = "onnx")] onnx: None };
        predict_classification(&calib, reader, writer, cfg).unwrap();
    }
    let s = String::from_utf8(out_buf.clone()).unwrap();
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(lines.len(), 1);
    let rec: ClassOutFull = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(rec.set_indices, vec![0, 1]);
    assert_eq!(rec.set_labels.as_deref(), Some(&["a".to_string(), "b".to_string()][..]));
    let sp = rec.set_probs.as_deref().unwrap();
    assert_eq!(sp.len(), 2);
    assert!((sp[0] - 0.6).abs() < 1e-12);
    assert!((sp[1] - 0.3).abs() < 1e-12);
    assert_eq!(rec.set_size, 2);
    assert_eq!(rec.max_prob_label.as_deref(), Some("a"));
    assert_eq!(rec.max_prob_index, Some(0));

    let reader = BufReader::new(Cursor::new(serde_json::json!({"probs": [0.6, 0.3, 0.1]}).to_string()));
    let mut out_buf2: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf2);
        let cfg = PredConfig { max_set_size: Some(1), include_probs: true, max_rows: None, #[cfg(feature = "onnx")] onnx: None };
        predict_classification(&calib, reader, writer, cfg).unwrap();
    }
    let rec2: ClassOutFull = serde_json::from_str(String::from_utf8(out_buf2).unwrap().lines().next().unwrap()).unwrap();
    assert_eq!(rec2.set_indices, vec![0]);
    assert_eq!(rec2.set_labels.as_deref(), Some(&["a".to_string()][..]));
    let sp2 = rec2.set_probs.as_deref().unwrap();
    assert_eq!(sp2.len(), 1);
    assert!((sp2[0] - 0.6).abs() < 1e-12);
    assert_eq!(rec2.set_size, 1);
    assert_eq!(rec2.max_prob_label.as_deref(), Some("a"));
    assert_eq!(rec2.max_prob_index, Some(0));
}
