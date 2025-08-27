use crate::calibrator::CalibModel;
use crate::utils::{argmax, ensure_prob_vector, jsonl_ser, softmax};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

#[derive(Debug, Clone)]
pub struct PredConfig {
    pub max_set_size: Option<usize>,
    pub include_probs: bool,
    pub max_rows: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassPredRow {
    #[serde(default)]
    probs: Option<Vec<f64>>,
    #[serde(default)]
    logits: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegrPredRow { pub y_pred: f64 }

#[derive(Debug, Clone, Serialize)]
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
pub struct RegrPredOut {
    pub y_pred: f64,
    pub lower: f64,
    pub upper: f64,
    pub width: f64,
}

pub fn predict_classification<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
) -> Result<()> {
    if calib.task != "class" { anyhow::bail!("calib is not classification"); }
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

pub fn predict_regression<R: BufRead, W: Write>(
    calib: &CalibModel,
    reader: R,
    mut writer: W,
    cfg: PredConfig,
) -> Result<()> {
    if calib.task != "regr" { anyhow::bail!("calib is not regression"); }
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
