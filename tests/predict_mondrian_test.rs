use onnxconformal_rs::calibrator::CalibModel;
use onnxconformal_rs::predictor::{predict_classification, PredConfig};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Cursor};

#[derive(Debug, Deserialize)]
struct ClassOutFull {
    set_indices: Vec<usize>,
    set_labels: Option<Vec<String>>,
    set_size: usize,
}

/// Ensures predictor honors per-label (Mondrian) thresholds when present.
#[test]
fn test_predict_classification_uses_per_label_thresholds() {
    // global_q = 0.2 -> threshold 0.8
    // per_label_q overrides: q("a") = 0.05 -> threshold 0.95 (exclude a),
    //                        q("b") = 0.8  -> threshold 0.20 (include b)
    let mut plq = HashMap::new();
    plq.insert("a".to_string(), 0.05);
    plq.insert("b".to_string(), 0.8);
    let calib = CalibModel {
        task: "class".into(),
        alpha: 0.2,
        global_q: 0.2,
        per_label_q: Some(plq),
        labels: Some(vec!["a".into(), "b".into()]),
        n: 1,
    };

    let input = serde_json::json!({"probs": [0.70, 0.30]}).to_string();
    let reader = BufReader::new(Cursor::new(input));
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig {
            max_set_size: None,
            include_probs: true,
            max_rows: None,
            #[cfg(feature = "onnx")]
            onnx: None,
        };
        predict_classification(&calib, reader, writer, cfg).unwrap();
    }
    let s = String::from_utf8(out_buf).unwrap();
    let line = s.lines().next().unwrap();
    let rec: ClassOutFull = serde_json::from_str(line).unwrap();
    assert_eq!(rec.set_indices, vec![1]);
    assert_eq!(rec.set_labels.as_deref(), Some(&["b".to_string()][..]));
    assert_eq!(rec.set_size, 1);
}
