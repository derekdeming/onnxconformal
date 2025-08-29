use anyhow::{bail, Result};
use ordered_float::OrderedFloat;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::{BufRead, Write};

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() { return vec![]; }
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 { vec![0.0; logits.len()] } else { exps.into_iter().map(|e| e / sum).collect() }
}

pub fn conformal_quantile(sorted_scores: &[f64], alpha: f64) -> f64 {
    let n = sorted_scores.len();
    if n == 0 { return f64::NAN; }
    let mut k = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as isize - 1;
    if k < 0 { k = 0; }
    if k as usize >= n { k = (n - 1) as isize; }
    sorted_scores[k as usize]
}

pub fn safe_sort(mut xs: Vec<f64>) -> Vec<f64> {
    xs.retain(|v| v.is_finite());
    xs.sort_by_key(|&v| OrderedFloat(v));
    xs
}

pub fn jsonl_deser<T: DeserializeOwned, R: BufRead>(reader: R, max_rows: Option<usize>) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        if let Some(m) = max_rows { if i >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        let v: T = serde_json::from_str(&l)?;
        out.push(v);
    }
    Ok(out)
}

pub fn jsonl_ser<T: Serialize, W: Write>(mut writer: W, rec: &T) -> Result<()> {
    let s = serde_json::to_string(rec)?;
    writeln!(writer, "{s}")?;
    Ok(())
}

pub fn argmax(xs: &[f64]) -> Option<usize> {
    if xs.is_empty() { return None; }
    let (mut idx, mut best) = (0usize, f64::NEG_INFINITY);
    for (i, &v) in xs.iter().enumerate() {
        if v > best { best = v; idx = i; }
    }
    Some(idx)
}

pub fn ensure_prob_vector(mut p: Vec<f64>) -> Vec<f64> {
    if p.is_empty() { return p; }
    for v in p.iter_mut() {
        if !v.is_finite() || *v < 0.0 { *v = 0.0; }
    }
    let s: f64 = p.iter().sum();
    if s <= 0.0 { vec![1.0 / p.len() as f64; p.len()] } else { p.into_iter().map(|v| v / s).collect() }
}

pub fn label_to_index(label: &serde_json::Value, labels: &[String]) -> Result<usize> {
    match label {
        serde_json::Value::Number(n) => {
            let i = n.as_u64().ok_or_else(|| anyhow::anyhow!("label index must be non-negative integer"))? as usize;
            if i >= labels.len() { bail!("label index {} out of range {}", i, labels.len()); }
            Ok(i)
        }
        serde_json::Value::String(s) => labels.iter().position(|t| t == s).ok_or_else(|| anyhow::anyhow!("label '{}' not in labels", s)),
        _ => bail!("label must be number or string"),
    }
}

/// Resolves a label index from either an explicit `label_index` or a `label` value.
/// If `canon_labels` is provided, string labels are mapped; otherwise numeric labels are required.
pub fn resolve_label_index(
    label_index: Option<usize>,
    label: Option<&serde_json::Value>,
    k: usize,
    canon_labels: Option<&[String]>,
) -> Result<usize> {
    if let Some(i) = label_index {
        if i >= k { bail!("label_index {} out of range {}", i, k); }
        return Ok(i);
    }
    if let Some(lbl) = label {
        if let Some(canon) = canon_labels { return label_to_index(lbl, canon); }
        match lbl {
            serde_json::Value::Number(n) => Ok(n.as_u64().ok_or_else(|| anyhow::anyhow!("label number invalid"))? as usize),
            _ => bail!("string label provided without 'labels' list"),
        }
    } else {
        bail!("row missing label/label_index");
    }
}
