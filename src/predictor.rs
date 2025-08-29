use crate::calibrator::CalibModel;
use crate::utils::{argmax, ensure_prob_vector, jsonl_ser, softmax};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

#[derive(Debug, Clone)]
/// Configuration for prediction routines.
pub struct PredConfig {
    pub max_set_size: Option<usize>,
    pub include_probs: bool,
    pub max_rows: Option<usize>,
    #[cfg(feature = "onnx")]
    pub onnx: Option<crate::onnx::OnnxOptions>,
    #[cfg(all(feature = "onnx", feature = "text"))]
    pub text: Option<crate::text::TextOptions>,
}

#[derive(Debug, Clone, Deserialize)]
/// Input row for classification prediction from probabilities or logits.
pub struct ClassPredRow {
    #[serde(default)]
    probs: Option<Vec<f64>>,
    #[serde(default)]
    logits: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
/// Input row for regression prediction from a point estimate `y_pred`.
pub struct RegrPredRow { pub y_pred: f64 }

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for classification prediction (`x` features).
pub struct ClassPredRowOnnx { pub x: Vec<f32> }

#[cfg(all(feature = "onnx", feature = "text"))]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for classification prediction from raw text.
pub struct ClassPredRowOnnxText { pub text: String }

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for regression prediction (`x` features).
pub struct RegrPredRowOnnx { pub x: Vec<f32> }

#[cfg(all(feature = "onnx", feature = "text"))]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for regression prediction from raw text.
pub struct RegrPredRowOnnxText { pub text: String }

#[derive(Debug, Clone, Serialize)]
/// Output record for classification prediction sets.
pub struct ClassPredOut {
    pub set_indices: Vec<usize>,
    pub set_labels: Option<Vec<String>>,
    pub set_size: usize,
    pub max_prob_label: Option<String>,
    pub max_prob_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set_probs: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
/// Output record for regression prediction intervals.
pub struct RegrPredOut {
    pub y_pred: f64,
    pub lower: f64,
    pub upper: f64,
    pub width: f64,
}

/// Produces conformal prediction sets for classification.
pub fn predict_classification<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
) -> Result<()> {
    if calib.task != "class" { anyhow::bail!("calib is not classification"); }
    #[cfg(feature = "onnx")]
    if let Some(onnx) = cfg.onnx.clone() {
        #[cfg(feature = "text")]
        if let Some(text) = cfg.text.clone() {
            return predict_classification_onnx_text(calib, reader, writer, cfg, onnx, text);
        }
        return predict_classification_onnx(calib, reader, writer, cfg, onnx);
    }
    let label_names = calib.labels.clone();
    let q = calib.global_q;
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows { if count >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        count += 1;
        let r: ClassPredRow = serde_json::from_str(&l)?;
        let probs = match (r.probs, r.logits) {
            (Some(p), _) => ensure_prob_vector(p),
            (None, Some(l)) => ensure_prob_vector(softmax(&l)),
            _ => anyhow::bail!("row missing probs/logits"),
        };
        if probs.is_empty() { continue; }
        let mut set: Vec<usize> = probs.iter().enumerate().filter_map(|(i, &p)| if (1.0 - p) <= q { Some(i) } else { None }).collect();
        if let Some(cap) = cfg.max_set_size {
            if set.len() > cap {
                set.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                set.truncate(cap);
                set.sort_unstable();
            }
        }
        let max_idx = argmax(&probs);
        let set_labels = label_names.as_ref().map(|names| set.iter().map(|&i| names.get(i).cloned().unwrap_or_else(|| format!("{}", i))).collect::<Vec<_>>());
        let max_label = match (&label_names, max_idx) { (Some(names), Some(i)) => Some(names[i].clone()), _ => None };
        let set_probs = if cfg.include_probs { Some(set.iter().map(|&i| probs[i]).collect()) } else { None };
        let out = ClassPredOut { set_indices: set.clone(), set_labels, set_size: set.len(), max_prob_label: max_label, max_prob_index: max_idx, set_probs };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}

/// Produces conformal prediction intervals for regression.
pub fn predict_regression<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
) -> Result<()> {
    if calib.task != "regr" { anyhow::bail!("calib is not regression"); }
    #[cfg(feature = "onnx")]
    if let Some(onnx) = cfg.onnx.clone() {
        #[cfg(feature = "text")]
        if let Some(text) = cfg.text.clone() {
            return predict_regression_onnx_text(calib, reader, writer, cfg, onnx, text);
        }
        return predict_regression_onnx(calib, reader, writer, cfg, onnx);
    }
    let q = calib.global_q;
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows { if count >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        count += 1;
        let r: RegrPredRow = serde_json::from_str(&l)?;
        let lower = r.y_pred - q;
        let upper = r.y_pred + q;
        let out = RegrPredOut { y_pred: r.y_pred, lower, upper, width: upper - lower };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}

#[cfg(feature = "onnx")]
/// ONNX-backed variant of classification prediction.
fn predict_classification_onnx<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
    onnx: crate::onnx::OnnxOptions,
) -> Result<()> {
    use crate::onnx::OnnxRunner;
    let mut runner = OnnxRunner::new(&onnx)?;
    let label_names = calib.labels.clone();
    let q = calib.global_q;
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows { if count >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        count += 1;
        let r: ClassPredRowOnnx = serde_json::from_str(&l)?;
        let out = runner.infer_vec_f32(&r.x)?;
        let logits_f64: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        let probs = ensure_prob_vector(softmax(&logits_f64));
        if probs.is_empty() { continue; }
        let mut set: Vec<usize> = probs.iter().enumerate().filter_map(|(i, &p)| if (1.0 - p) <= q { Some(i) } else { None }).collect();
        if let Some(cap) = cfg.max_set_size { if set.len() > cap { set.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap()); set.truncate(cap); set.sort_unstable(); } }
        let max_idx = argmax(&probs);
        let set_labels = label_names.as_ref().map(|names| set.iter().map(|&i| names.get(i).cloned().unwrap_or_else(|| format!("{}", i))).collect::<Vec<_>>());
        let max_label = match (&label_names, max_idx) { (Some(names), Some(i)) => Some(names[i].clone()), _ => None };
        let set_probs = if cfg.include_probs { Some(set.iter().map(|&i| probs[i]).collect()) } else { None };
        let out = ClassPredOut { set_indices: set.clone(), set_labels, set_size: set.len(), max_prob_label: max_label, max_prob_index: max_idx, set_probs };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}

#[cfg(all(feature = "onnx", feature = "text"))]
/// ONNX-backed variant of classification prediction from raw text using a tokenizer.
fn predict_classification_onnx_text<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
    onnx: crate::onnx::OnnxOptions,
    text: crate::text::TextOptions,
) -> Result<()> {
    use crate::onnx::OnnxRunner;
    use crate::text::TextTokenizer;
    let mut runner = OnnxRunner::new(&onnx)?;
    let tok = TextTokenizer::new(&text)?;
    let label_names = calib.labels.clone();
    let q = calib.global_q;
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows { if count >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        count += 1;
        let r: ClassPredRowOnnxText = serde_json::from_str(&l)?;
        let ids = tok.encode_ids_i64(&r.text)?;
        let out = runner.infer_vec_i64(&ids)?;
        let logits_f64: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        let probs = ensure_prob_vector(softmax(&logits_f64));
        if probs.is_empty() { continue; }
        let mut set: Vec<usize> = probs.iter().enumerate().filter_map(|(i, &p)| if (1.0 - p) <= q { Some(i) } else { None }).collect();
        if let Some(cap) = cfg.max_set_size { if set.len() > cap { set.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap()); set.truncate(cap); set.sort_unstable(); } }
        let max_idx = argmax(&probs);
        let set_labels = label_names.as_ref().map(|names| set.iter().map(|&i| names.get(i).cloned().unwrap_or_else(|| format!("{}", i))).collect::<Vec<_>>());
        let max_label = match (&label_names, max_idx) { (Some(names), Some(i)) => Some(names[i].clone()), _ => None };
        let set_probs = if cfg.include_probs { Some(set.iter().map(|&i| probs[i]).collect()) } else { None };
        let out = ClassPredOut { set_indices: set.clone(), set_labels, set_size: set.len(), max_prob_label: max_label, max_prob_index: max_idx, set_probs };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}

#[cfg(feature = "onnx")]
/// ONNX-backed variant of regression prediction.
fn predict_regression_onnx<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
    onnx: crate::onnx::OnnxOptions,
) -> Result<()> {
    use crate::onnx::OnnxRunner;
    let mut runner = OnnxRunner::new(&onnx)?;
    let q = calib.global_q;
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows { if count >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        count += 1;
        let r: RegrPredRowOnnx = serde_json::from_str(&l)?;
        let out = runner.infer_vec_f32(&r.x)?;
        if out.is_empty() { continue; }
        let y_pred = out[0] as f64;
        let lower = y_pred - q;
        let upper = y_pred + q;
        let out = RegrPredOut { y_pred, lower, upper, width: upper - lower };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}

#[cfg(all(feature = "onnx", feature = "text"))]
/// ONNX-backed variant of regression prediction from raw text using a tokenizer.
fn predict_regression_onnx_text<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
    onnx: crate::onnx::OnnxOptions,
    text: crate::text::TextOptions,
) -> Result<()> {
    use crate::onnx::OnnxRunner;
    use crate::text::TextTokenizer;
    let mut runner = OnnxRunner::new(&onnx)?;
    let tok = TextTokenizer::new(&text)?;
    let q = calib.global_q;
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows { if count >= m { break; } }
        let l = line?;
        if l.trim().is_empty() { continue; }
        count += 1;
        let r: RegrPredRowOnnxText = serde_json::from_str(&l)?;
        let ids = tok.encode_ids_i64(&r.text)?;
        let out = runner.infer_vec_i64(&ids)?;
        if out.is_empty() { continue; }
        let y_pred = out[0] as f64;
        let lower = y_pred - q;
        let upper = y_pred + q;
        let out = RegrPredOut { y_pred, lower, upper, width: upper - lower };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}
