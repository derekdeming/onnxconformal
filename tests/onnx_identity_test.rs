#![cfg(feature = "onnx")]

use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel};
use onnxconformal_rs::predictor::{predict_classification, predict_regression, PredConfig};
use std::fs;
use std::io::Write;
use std::io::{BufReader, BufWriter, Cursor};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// identity.ort from ort v2.0.0-rc.10, embedded as base64.
const IDENTITY_ORT_B64: &str =
    "FAAAAE9SVE0MABAADAAIAAAABAAMAAAADAAAAJQAAADgAwAA0v3//wQAAAABAAAABAAAANj///8QAAAABAAAAAEAAAAgAAAADAAAADpJZGVudGl0eToyMQAAAAAIAAwABAAIAAgAAAAIAAAADAAAAAEAAABWAAAAAgAAABwAAAAMAAAAAAAGAAgABwAGAAAAAAAAAQQABAAEAAAAFAAkABgAFAAAAAAAEAAIAAAABAAUAAAANAAAAP////////9/OAMAAMwBAAAKAAAAAAAAAAAAAAAUACQAIAAcABgAFAAQAAwACAAEABQAAAB4AQAAeAEAAIgBAAAUAAAAAQAAABQAAADkAAAAWAEAAAEAAAAYAAAAAQAAAEQAAAAAAAoADAAAAAQACAAKAAAADAAAAAQAAAAAAAAAAAAAAAAAHgAwACwAKAAkACAAAAAcAAAAGAAUABAADAAIAAQAHgAAADAAAAAwAAAAJAAAADAAAAA0AAAAOAAAAFAAAAAVAAAAeAIAAFQAAABYAAAAAAAAAAAAAAABAAAAAQAAAAEAAADQAAAAAQAAANwAAAAUAAAAQ1BVRXhlY3V0aW9uUHJvdmlkZXIAAAAACAAAAElkZW50aXR5AAAAAAAAAAAAAAAAAAAAAAAAAAACAAAARAAAABAAAAAAAAoADAAIAAAABAAKAAAACAAAAGgAAADO////DAAAAAABBgAKAAQABgAAAAkAAAAAAAoADgAIAAAABAAKAAAAFAAAAEwAAAAAAAoACgAAAAkABAAKAAAADAAAAAABBgAIAAQABgAAAAkAAAAAAAAAAAAAAAEAAAAEAAAABgAAAG91dHB1dAAAAQAAAAQAAAAFAAAAaW5wdXQAAAAIAAAAUAEAABwBAADwAAAAxAAAAJAAAABkAAAAMAAAAAQAAAAI////AQAAAAAAAAAEAAAAEAAAAG9yZy5weXRvcmNoLmF0ZW4AAAAAMP///wEAAAAAAAAABAAAABoAAABjb20ubWljcm9zb2Z0LmV4cGVyaW1lbnRhbAAAMP///wEAAAAAAAAAAAAAAAQAAAANAAAAY29tLm1pY3Jvc29mdAAAAIj///8BAAAAAAAAAAQAAAAYAAAAYWkub25ueC5wcmV2aWV3LnRyYWluaW5nAAAAALj///8BAAAAAAAAAAQAAAAQAAAAYWkub25ueC50cmFpbmluZwAAAADg////BQAAAAAAAAAEAAAACgAAAGFpLm9ubngubWwAAAgAEAAMAAQACAAAAAEAAAAAAAAABAAAABMAAABjb20ubWljcm9zb2Z0Lm5jaHdjAAgAFAAQAAQACAAAABUAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAEAAAA2AAAA";

/// Minimal base64 decoder for the embedded ORT model.
fn decode_b64(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    let mut i = 0;
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(64),
            _ => None,
        }
    }
    for &b in s.as_bytes() {
        if b == b'\n' || b == b'\r' || b == b' ' { continue; }
        if let Some(v) = val(b) {
            buf[i] = v;
            i += 1;
            if i == 4 {
                let a = buf[0]; let b = buf[1]; let c = buf[2]; let d = buf[3];
                if a == 64 || b == 64 { break; }
                out.push((a << 2) | (b >> 4));
                if c != 64 { out.push(((b & 0xF) << 4) | (c >> 2)); }
                if d != 64 { out.push(((c & 0x3) << 6) | d); }
                i = 0;
            }
        }
    }
    out
}

/// Creates a unique temporary file path for test artifacts.
fn tmp_file(name: &str, ext: &str) -> PathBuf {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("onnxconformal_{}_{}.{}", name, ts, ext));
    p
}

/// Decodes and writes the embedded identity model to a temp file.
fn write_identity_model() -> PathBuf {
    let bytes = decode_b64(IDENTITY_ORT_B64);
    assert!(!bytes.is_empty());
    let path = tmp_file("identity", "ort");
    fs::write(&path, bytes).unwrap();
    path
}

/// ONNX classification: calibrate from features and predict using identity model.
#[test]
fn onnx_classification_identity_end_to_end() {
    let model_path = write_identity_model();
    let calib_path = tmp_file("class_feats", "jsonl");
    {
        let mut f = fs::File::create(&calib_path).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[2.0, 0.0], "labels":["a","b"], "label":"a"}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[-1.0, 1.0], "label":"b"}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[1.5, 0.5], "label_index": 0}).to_string()).unwrap();
    }

    let cfg = CalibConfig {
        alpha: 0.1,
        mondrian: false,
        max_rows: None,
        onnx: Some(onnxconformal_rs::onnx::OnnxOptions { model: model_path.to_string_lossy().to_string(), input_name: None, output_name: None })
    };
    let model = CalibModel::fit_from_file(calib_path.to_str().unwrap(), CalibFileKind::Classification, cfg).unwrap();
    assert_eq!(model.task, "class");
    assert_eq!(model.labels.as_deref(), Some(&["a".to_string(), "b".to_string()][..]));

    let pred_path = tmp_file("class_pred_feats", "jsonl");
    {
        let mut f = fs::File::create(&pred_path).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[0.6, 0.3]}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[0.2, 0.8]}).to_string()).unwrap();
    }
    let reader = BufReader::new(fs::File::open(&pred_path).unwrap());
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig { max_set_size: Some(1), include_probs: true, max_rows: None, onnx: Some(onnxconformal_rs::onnx::OnnxOptions { model: model_path.to_string_lossy().to_string(), input_name: None, output_name: None }) };
        predict_classification(&model, reader, writer, cfg).unwrap();
    }
    let s = String::from_utf8(out_buf).unwrap();
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(lines.len(), 2);
    #[derive(serde::Deserialize)]
    struct Out { set_indices: Vec<usize>, set_size: usize, max_prob_index: Option<usize> }
    let o1: Out = serde_json::from_str(lines[0]).unwrap();
    let o2: Out = serde_json::from_str(lines[1]).unwrap();
    assert!(o1.set_size <= 1);
    assert!(o2.set_size <= 1);
}

/// ONNX regression: calibrate from features + y_true and predict intervals.
#[test]
fn onnx_regression_identity_end_to_end() {
    let model_path = write_identity_model();
    let calib_path = tmp_file("regr_feats", "jsonl");
    {
        let mut f = fs::File::create(&calib_path).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[1.2], "y_true": 1.0}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[-0.1], "y_true": 0.0}).to_string()).unwrap();
        writeln!(f, "{}", serde_json::json!({"x":[-0.7], "y_true": -1.0}).to_string()).unwrap();
    }
    let cfg = CalibConfig { alpha: 0.2, mondrian: false, max_rows: None, onnx: Some(onnxconformal_rs::onnx::OnnxOptions { model: model_path.to_string_lossy().to_string(), input_name: None, output_name: None }) };
    let model = CalibModel::fit_from_file(calib_path.to_str().unwrap(), CalibFileKind::Regression, cfg).unwrap();
    assert_eq!(model.task, "regr");
    assert!(model.global_q.is_finite());

    let preds = [serde_json::json!({"x":[1.0]}), serde_json::json!({"x":[-2.0]})]
        .into_iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n");
    let reader = BufReader::new(Cursor::new(preds));
    let mut out_buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut out_buf);
        let cfg = PredConfig { max_set_size: None, include_probs: false, max_rows: None, onnx: Some(onnxconformal_rs::onnx::OnnxOptions { model: model_path.to_string_lossy().to_string(), input_name: None, output_name: None }) };
        predict_regression(&model, reader, writer, cfg).unwrap();
    }
    let s = String::from_utf8(out_buf).unwrap();
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(lines.len(), 2);
}
