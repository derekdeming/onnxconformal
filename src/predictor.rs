use crate::calibrator::CalibModel;
use crate::utils::{argmax, ensure_prob_vector, jsonl_ser, softmax};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::collections::HashMap;

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
pub struct RegrPredRow {
    pub y_pred: f64,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for classification prediction (`x` features).
pub struct ClassPredRowOnnx {
    pub x: Vec<f32>,
}

#[cfg(all(feature = "onnx", feature = "text"))]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for classification prediction from raw text.
pub struct ClassPredRowOnnxText {
    pub text: String,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for regression prediction (`x` features).
pub struct RegrPredRowOnnx {
    pub x: Vec<f32>,
}

#[cfg(all(feature = "onnx", feature = "text"))]
#[derive(Debug, Clone, Deserialize)]
/// ONNX-backed input row for regression prediction from raw text.
pub struct RegrPredRowOnnxText {
    pub text: String,
}

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

fn build_class_output(
    probs: &[f64],
    label_names: Option<&Vec<String>>,
    q: f64,
    max_set_size: Option<usize>,
    include_probs: bool,
) -> ClassPredOut {
    // Helper to resolve per-label thresholds (Mondrian). Falls back to global q.
    let per_label_q: Option<&HashMap<String, f64>> = None;
    build_class_output_with_mondrian(probs, label_names, q, per_label_q, max_set_size, include_probs)
}

fn build_class_output_with_mondrian(
    probs: &[f64],
    label_names: Option<&Vec<String>>,
    global_q: f64,
    per_label_q: Option<&HashMap<String, f64>>,
    max_set_size: Option<usize>,
    include_probs: bool,
) -> ClassPredOut {
    let mut set: Vec<usize> = Vec::new();
    for (i, &p) in probs.iter().enumerate() {
        let q_i = if let Some(map) = per_label_q {
            let key = match label_names {
                Some(names) => names.get(i).cloned().unwrap_or_else(|| format!("{}", i)),
                None => format!("{}", i),
            };
            *map.get(&key).unwrap_or(&global_q)
        } else {
            global_q
        };
        if (1.0 - p) <= q_i {
            set.push(i);
        }
    }
    if let Some(cap) = max_set_size {
        if set.len() > cap {
            set.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            set.truncate(cap);
            set.sort_unstable();
        }
    }
    let max_idx = argmax(probs);
    let set_labels = label_names.map(|names| {
        set.iter()
            .map(|&i| names.get(i).cloned().unwrap_or_else(|| format!("{}", i)))
            .collect::<Vec<_>>()
    });
    let max_label = match (label_names, max_idx) {
        (Some(names), Some(i)) => Some(names[i].clone()),
        _ => None,
    };
    let set_probs = if include_probs {
        Some(set.iter().map(|&i| probs[i]).collect())
    } else {
        None
    };
    ClassPredOut {
        set_indices: set.clone(),
        set_labels,
        set_size: set.len(),
        max_prob_label: max_label,
        max_prob_index: max_idx,
        set_probs,
    }
}

/// Produces conformal prediction sets for classification.
pub fn predict_classification<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
) -> Result<()> {
    if calib.task != "class" {
        anyhow::bail!("calib is not classification");
    }
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
    let per_label_q = calib.per_label_q.as_ref();
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows {
            if count >= m {
                break;
            }
        }
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        count += 1;
        let r: ClassPredRow = serde_json::from_str(&l)?;
        let probs = match (r.probs, r.logits) {
            (Some(p), _) => ensure_prob_vector(p),
            (None, Some(l)) => ensure_prob_vector(softmax(&l)),
            _ => anyhow::bail!("row missing probs/logits"),
        };
        if probs.is_empty() {
            continue;
        }
        let out = build_class_output_with_mondrian(
            &probs,
            label_names.as_ref(),
            q,
            per_label_q,
            cfg.max_set_size,
            cfg.include_probs,
        );
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
    if calib.task != "regr" {
        anyhow::bail!("calib is not regression");
    }
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
        if let Some(m) = cfg.max_rows {
            if count >= m {
                break;
            }
        }
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        count += 1;
        let r: RegrPredRow = serde_json::from_str(&l)?;
        let lower = r.y_pred - q;
        let upper = r.y_pred + q;
        let out = RegrPredOut {
            y_pred: r.y_pred,
            lower,
            upper,
            width: upper - lower,
        };
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
    let per_label_q = calib.per_label_q.as_ref();
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows {
            if count >= m {
                break;
            }
        }
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        count += 1;
        let r: ClassPredRowOnnx = serde_json::from_str(&l)?;
        let out = runner.infer_vec_f32(&r.x)?;
        let logits_f64: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        let probs = ensure_prob_vector(softmax(&logits_f64));
        if probs.is_empty() {
            continue;
        }
        let out = build_class_output_with_mondrian(
            &probs,
            label_names.as_ref(),
            q,
            per_label_q,
            cfg.max_set_size,
            cfg.include_probs,
        );
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
    let per_label_q = calib.per_label_q.as_ref();
    let mut count = 0usize;
    for line in reader.lines() {
        if let Some(m) = cfg.max_rows {
            if count >= m {
                break;
            }
        }
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        count += 1;
        let r: ClassPredRowOnnxText = serde_json::from_str(&l)?;
        let (ids, mask, type_ids) = tok.encode_with_aux_i64(&r.text)?;
        let out = if let Some(names) = &onnx.input_names {
            match names.as_slice() {
                [id] => runner.infer_i64_named(&[(id.as_str(), &ids)])?,
                [id, maskn] => runner.infer_i64_named(&[(id.as_str(), &ids), (maskn.as_str(), &mask)])?,
                [id, maskn, typen] => runner.infer_i64_named(&[(id.as_str(), &ids), (maskn.as_str(), &mask), (typen.as_str(), &type_ids)])?,
                _ => anyhow::bail!("expected 1-3 --onnx-inputs for text models (input_ids[,attention_mask[,token_type_ids]])"),
            }
        } else if let Some(single) = onnx.input_name.as_ref() {
            runner.infer_i64_named(&[(single.as_str(), &ids)])?
        } else {
            anyhow::bail!("text mode requires --onnx-input or --onnx-inputs");
        };
        let logits_f64: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        let probs = ensure_prob_vector(softmax(&logits_f64));
        if probs.is_empty() {
            continue;
        }
        let out = build_class_output_with_mondrian(
            &probs,
            label_names.as_ref(),
            q,
            per_label_q,
            cfg.max_set_size,
            cfg.include_probs,
        );
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
        if let Some(m) = cfg.max_rows {
            if count >= m {
                break;
            }
        }
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        count += 1;
        let r: RegrPredRowOnnx = serde_json::from_str(&l)?;
        let out = runner.infer_vec_f32(&r.x)?;
        if out.is_empty() {
            continue;
        }
        let y_pred = out[0] as f64;
        let lower = y_pred - q;
        let upper = y_pred + q;
        let out = RegrPredOut {
            y_pred,
            lower,
            upper,
            width: upper - lower,
        };
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
        if let Some(m) = cfg.max_rows {
            if count >= m {
                break;
            }
        }
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        count += 1;
        let r: RegrPredRowOnnxText = serde_json::from_str(&l)?;
        let (ids, mask, type_ids) = tok.encode_with_aux_i64(&r.text)?;
        let out = if let Some(names) = &onnx.input_names {
            match names.as_slice() {
                [id] => runner.infer_i64_named(&[(id.as_str(), &ids)])?,
                [id, maskn] => {
                    runner.infer_i64_named(&[(id.as_str(), &ids), (maskn.as_str(), &mask)])?
                }
                [id, maskn, typen] => runner.infer_i64_named(&[
                    (id.as_str(), &ids),
                    (maskn.as_str(), &mask),
                    (typen.as_str(), &type_ids),
                ])?,
                _ => anyhow::bail!("expected 1-3 --onnx-inputs for text models"),
            }
        } else if let Some(single) = onnx.input_name.as_ref() {
            runner.infer_i64_named(&[(single.as_str(), &ids)])?
        } else {
            anyhow::bail!("text mode requires --onnx-input or --onnx-inputs");
        };
        if out.is_empty() {
            continue;
        }
        let y_pred = out[0] as f64;
        let lower = y_pred - q;
        let upper = y_pred + q;
        let out = RegrPredOut {
            y_pred,
            lower,
            upper,
            width: upper - lower,
        };
        jsonl_ser(&mut writer, &out)?;
    }
    Ok(())
}
