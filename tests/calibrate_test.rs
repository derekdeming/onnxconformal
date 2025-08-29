use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel};
use onnxconformal_rs::utils::conformal_quantile;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp_file(name: &str) -> PathBuf {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("onnxconformal_test_{}_{}.jsonl", name, ts));
    p
}

#[test]
fn test_calibrate_classification() {
    let path = tmp_file("class");
    {
        let mut f = fs::File::create(&path).unwrap();
        // probs and label_index
        writeln!(f, "{}", serde_json::json!({"probs": [0.6, 0.4], "label_index": 0}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"probs": [0.2, 0.8], "label_index": 1}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"probs": [0.7, 0.3], "label_index": 0}).to_string()).unwrap();
    }

    let cfg = CalibConfig { alpha: 0.1, mondrian: false, max_rows: None, #[cfg(feature = "onnx")] onnx: None };
    let model = CalibModel::fit_from_file(path.to_str().unwrap(), CalibFileKind::Classification, cfg).unwrap();
    assert_eq!(model.task, "class");
    assert_eq!(model.alpha, 0.1);
    assert_eq!(model.per_label_q, None);
    assert_eq!(model.n, 3);

    // compute expected q
    let p_true = vec![0.6, 0.8, 0.7];
    let scores: Vec<f64> = p_true.into_iter().map(|p| 1.0 - p).collect();
    let sorted = {
        let mut s = scores.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    };
    let q_exp = conformal_quantile(&sorted, 0.1);
    assert!((model.global_q - q_exp).abs() < 1e-12);

    let _ = fs::remove_file(&path);
}

#[test]
fn test_calibrate_regression() {
    let path = tmp_file("regr");
    {
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "{}", serde_json::json!({"y_true": 1.0, "y_pred": 1.2}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"y_true": 0.0, "y_pred": -0.1}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"y_true": -1.0, "y_pred": -0.7}).to_string()).unwrap();
    }

    let cfg = CalibConfig { alpha: 0.2, mondrian: false, max_rows: None, #[cfg(feature = "onnx")] onnx: None };
    let model = CalibModel::fit_from_file(path.to_str().unwrap(), CalibFileKind::Regression, cfg).unwrap();
    assert_eq!(model.task, "regr");
    assert_eq!(model.alpha, 0.2);
    assert!(model.per_label_q.is_none());
    assert_eq!(model.labels, None);
    assert_eq!(model.n, 3);
    assert!(model.global_q.is_finite());

    let _ = fs::remove_file(&path);
}
