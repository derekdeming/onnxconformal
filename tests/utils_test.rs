use onnxconformal_rs::utils::{softmax, ensure_prob_vector, conformal_quantile, argmax, jsonl_deser, jsonl_ser};
use serde::{Serialize, Deserialize};
use std::io::{BufReader, BufWriter, Cursor};

#[test]
fn test_softmax_basic() {
    let v = softmax(&[0.0, 0.0]);
    assert_eq!(v.len(), 2);
    let s: f64 = v.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
    assert!((v[0] - 0.5).abs() < 1e-12);
    assert!((v[1] - 0.5).abs() < 1e-12);
}

#[test]
fn test_ensure_prob_vector_normalizes_and_clips() {
    let out = ensure_prob_vector(vec![1.0, -1.0, f64::NAN, 1.0]);
    assert_eq!(out.len(), 4);
    let s: f64 = out.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
    assert!(out.iter().all(|p| *p >= 0.0));
}

#[test]
fn test_conformal_quantile_indexing() {
    let xs = vec![0.1, 0.2, 0.3, 0.4];
    let q = conformal_quantile(&xs, 0.1);
    assert!((q - 0.4).abs() < 1e-12);
}

#[test]
fn test_argmax() {
    let idx = argmax(&[0.1, 2.0, 1.5]).unwrap();
    assert_eq!(idx, 1);
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Row { a: i32 }

#[test]
fn test_jsonl_roundtrip() {
    let mut buf: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut buf);
        jsonl_ser(writer, &Row { a: 1 }).unwrap();
    }
    let cursor = Cursor::new(buf);
    let reader = BufReader::new(cursor);
    let rows: Vec<Row> = jsonl_deser(reader, None).unwrap();
    assert_eq!(rows, vec![Row { a: 1 }]);
}

