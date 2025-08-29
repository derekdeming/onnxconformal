use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel};
use onnxconformal_rs::nonconformity::class_score;
use onnxconformal_rs::utils::{conformal_quantile, safe_sort, softmax};
use std::fs;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Creates a unique temporary file path for test artifacts.
fn tmp_file(name: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "onnxconformal_class_mondrian_{}_{}.jsonl",
        name, ts
    ));
    p
}

/// Fits calibration from logits with string labels (Mondrian) and
/// verifies the global conformal quantile matches a recomputed value.
#[test]
fn test_calibrate_classification_logits_and_labels_mondrian() {
    let path = tmp_file("class_logits_labels");
    {
        let mut f = fs::File::create(&path).unwrap();
        writeln!(
            f,
            "{}",
            serde_json::json!({"labels":["ham","phish"], "logits":[2.0, 0.0], "label":"ham"})
        )
        .unwrap();
        writeln!(
            f,
            "{}",
            serde_json::json!({"logits":[-1.0, 1.0], "label":"phish"})
        )
        .unwrap();
        writeln!(
            f,
            "{}",
            serde_json::json!({"logits":[1.5, 0.5], "label":"ham"})
        )
        .unwrap();
    }

    let cfg = CalibConfig {
        alpha: 0.2,
        mondrian: true,
        max_rows: None,
        #[cfg(feature = "onnx")]
        onnx: None,
    };
    let model =
        CalibModel::fit_from_file(path.to_str().unwrap(), CalibFileKind::Classification, cfg)
            .unwrap();
    assert_eq!(model.task, "class");
    assert_eq!(model.alpha, 0.2);
    assert_eq!(
        model.labels.as_deref(),
        Some(&["ham".to_string(), "phish".to_string()][..])
    );
    assert!(model.per_label_q.is_some());

    let logits = vec![
        (vec![2.0, 0.0], 0usize),
        (vec![-1.0, 1.0], 1usize),
        (vec![1.5, 0.5], 0usize),
    ];
    let mut scores = Vec::new();
    for (lg, li) in logits {
        let p = softmax(&lg);
        scores.push(class_score(p[li]));
    }
    let sorted = safe_sort(scores);
    let q_exp = conformal_quantile(&sorted, 0.2);
    assert!((model.global_q - q_exp).abs() < 1e-12);

    let _ = fs::remove_file(&path);
}
