use crate::nonconformity::{class_score, regr_score};
use crate::utils::{conformal_quantile, ensure_prob_vector, jsonl_deser, label_to_index, safe_sort, softmax};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone)]
/// Configuration for calibration from JSONL data.
pub struct CalibConfig {
    pub alpha: f64,
    pub mondrian: bool,
    pub max_rows: Option<usize>,
    #[cfg(feature = "onnx")]
    pub onnx: Option<crate::onnx::OnnxOptions>,
}

#[derive(Debug, Clone, Copy)]
/// Input file kind for calibration.
pub enum CalibFileKind { Classification, Regression }

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Persisted calibration model with global and optional per‑label thresholds.
pub struct CalibModel {
    pub task: String,
    pub alpha: f64,
    pub global_q: f64,
    pub per_label_q: Option<HashMap<String, f64>>,
    pub labels: Option<Vec<String>>,
    pub n: usize,
}

impl CalibModel {
    /// Fits a calibration model from a JSONL file.
    pub fn fit_from_file(path: &str, kind: CalibFileKind, cfg: CalibConfig) -> Result<Self> {
        match kind {
            CalibFileKind::Classification => {
                #[cfg(feature = "onnx")]
                if let Some(onnx) = cfg.onnx.clone() { return fit_class_from_jsonl_onnx(path, cfg, onnx); }
                fit_class_from_jsonl(path, cfg)
            }
            CalibFileKind::Regression => {
                #[cfg(feature = "onnx")]
                if let Some(onnx) = cfg.onnx.clone() { return fit_regr_from_jsonl_onnx(path, cfg, onnx); }
                fit_regr_from_jsonl(path, cfg)
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let w = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(w, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let r = std::fs::File::open(path)?;
        let m: CalibModel = serde_json::from_reader(r)?;
        Ok(m)
    }
}

#[derive(Debug, Clone, Serialize)]
/// Human‑readable summary of a calibration model.
pub struct CalibSummary {
    pub task: String,
    pub alpha: f64,
    pub n: usize,
    pub global_q: f64,
    pub labels: Option<Vec<String>>,
    pub per_label_q: Option<HashMap<String, f64>>,
}

impl From<&CalibModel> for CalibSummary {
    fn from(m: &CalibModel) -> Self {
        Self {
            task: m.task.clone(),
            alpha: m.alpha,
            n: m.n,
            global_q: m.global_q,
            labels: m.labels.clone(),
            per_label_q: m.per_label_q.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ClassCalibRow {
    /// Either "probs" or "logits" must be present
    #[serde(default)]
    probs: Option<Vec<f64>>,
    #[serde(default)]
    logits: Option<Vec<f64>>,

    /// True label: can be "label": "spam" or "label_index": 1
    #[serde(default)]
    label: Option<serde_json::Value>,
    #[serde(default)]
    label_index: Option<usize>,

    /// Optional canonical label list for mapping strings -> indices
    #[serde(default)]
    labels: Option<Vec<String>>,
}

/// Fits a classification calibration model from probabilities/logits JSONL.
fn fit_class_from_jsonl(path: &str, cfg: CalibConfig) -> Result<CalibModel> {
    let reader = BufReader::new(File::open(path).with_context(|| "open class calib file")?);
    let rows: Vec<ClassCalibRow> = crate::utils::jsonl_deser(reader, cfg.max_rows)?;
    if rows.is_empty() { anyhow::bail!("no calibration rows"); }

    let mut canon_labels: Option<Vec<String>> = None;
    for r in rows.iter() {
        if let Some(ls) = &r.labels {
            if !ls.is_empty() { canon_labels = Some(ls.clone()); }
        }
    }

    let mut scores: Vec<f64> = Vec::with_capacity(rows.len());
    let mut per_label: HashMap<usize, Vec<f64>> = HashMap::new();

    for r in rows.iter() {
        let p = match (&r.probs, &r.logits) {
            (Some(ps), _) => ensure_prob_vector(ps.clone()),
            (None, Some(lg)) => {
                let sm = softmax(lg);
                ensure_prob_vector(sm)
            }
            _ => anyhow::bail!("row missing probs/logits"),
        };
        let k = p.len();
        if k == 0 { continue; }

        let idx = if let Some(i) = r.label_index {
            if i >= k { anyhow::bail!("label_index {} out of range {}", i, k); }
            i
        } else if let Some(lbl) = &r.label {
            if let Some(canon) = &canon_labels {
                label_to_index(lbl, canon)?
            } else {
                match lbl {
                    serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| anyhow::anyhow!("label number invalid"))? as usize,
                    _ => anyhow::bail!("string label provided without 'labels' list"),
                }
            }
        } else {
            anyhow::bail!("row missing label/label_index");
        };

        let s = class_score(p[idx]);
        scores.push(s);
        if cfg.mondrian {
            per_label.entry(idx).or_default().push(s);
        }
    }

    let sorted = crate::utils::safe_sort(scores);
    if sorted.is_empty() { anyhow::bail!("no valid scores for calibration"); }
    let q = crate::utils::conformal_quantile(&sorted, cfg.alpha);

    let per_label_q = if cfg.mondrian {
        let mut m = HashMap::new();
        for (li, v) in per_label.into_iter() {
            let vv = crate::utils::safe_sort(v);
            if vv.is_empty() { continue; }
            let qq = crate::utils::conformal_quantile(&vv, cfg.alpha);
            let key = if let Some(names) = &canon_labels { names[li].clone() } else { format!("{}", li) };
            m.insert(key, qq);
        }
        Some(m)
    } else { None };

    Ok(CalibModel {
        task: "class".into(),
        alpha: cfg.alpha,
        global_q: q,
        per_label_q,
        labels: canon_labels,
        n: sorted.len(),
    })
}

#[derive(Debug, Clone, Deserialize)]
struct RegrCalibRow {
    y_true: f64,
    y_pred: f64,
}

/// Fits a regression calibration model from `y_true/y_pred` JSONL.
fn fit_regr_from_jsonl(path: &str, cfg: CalibConfig) -> Result<CalibModel> {
    let reader = BufReader::new(File::open(path).with_context(|| "open regr calib file")?);
    let rows: Vec<RegrCalibRow> = jsonl_deser(reader, cfg.max_rows)?;
    if rows.is_empty() { anyhow::bail!("no calibration rows"); }

    let mut scores = Vec::with_capacity(rows.len());
    for r in rows {
        let s = regr_score(r.y_true, r.y_pred);
        if s.is_finite() { scores.push(s); }
    }
    let sorted = safe_sort(scores);
    if sorted.is_empty() { anyhow::bail!("no valid scores for calibration"); }
    let q = conformal_quantile(&sorted, cfg.alpha);

    Ok(CalibModel {
        task: "regr".into(),
        alpha: cfg.alpha,
        global_q: q,
        per_label_q: None,
        labels: None,
        n: sorted.len(),
    })
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Deserialize)]
struct ClassCalibRowOnnx {
    x: Vec<f32>,
    #[serde(default)]
    label: Option<serde_json::Value>,
    #[serde(default)]
    label_index: Option<usize>,
    #[serde(default)]
    labels: Option<Vec<String>>,
}

#[cfg(feature = "onnx")]
/// Fits a classification calibration model by running an ONNX model over
/// feature vectors (`x`) and using the output as logits.
fn fit_class_from_jsonl_onnx(path: &str, cfg: CalibConfig, onnx: crate::onnx::OnnxOptions) -> Result<CalibModel> {
    use crate::onnx::OnnxRunner;
    let reader = BufReader::new(File::open(path).with_context(|| "open class calib file (onnx)")?);
    let rows: Vec<ClassCalibRowOnnx> = crate::utils::jsonl_deser(reader, cfg.max_rows)?;
    if rows.is_empty() { anyhow::bail!("no calibration rows"); }

    let mut canon_labels: Option<Vec<String>> = None;
    for r in rows.iter() {
        if let Some(ls) = &r.labels {
            if !ls.is_empty() { canon_labels = Some(ls.clone()); }
        }
    }

    let mut runner = OnnxRunner::new(&onnx)?;
    let mut scores: Vec<f64> = Vec::with_capacity(rows.len());
    let mut per_label: HashMap<usize, Vec<f64>> = HashMap::new();

    for r in rows.iter() {
        let out = runner.infer_vec_f32(&r.x)?;
        if out.is_empty() { anyhow::bail!("onnx model produced empty output"); }
        let logits_f64: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        let probs = ensure_prob_vector(softmax(&logits_f64));
        let k = probs.len();
        let idx = if let Some(i) = r.label_index {
            if i >= k { anyhow::bail!("label_index {} out of range {}", i, k); }
            i
        } else if let Some(lbl) = &r.label {
            if let Some(canon) = &canon_labels {
                label_to_index(lbl, canon)?
            } else {
                match lbl {
                    serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| anyhow::anyhow!("label number invalid"))? as usize,
                    _ => anyhow::bail!("string label provided without 'labels' list"),
                }
            }
        } else {
            anyhow::bail!("row missing label/label_index");
        };
        let s = class_score(probs[idx]);
        scores.push(s);
        if cfg.mondrian {
            per_label.entry(idx).or_default().push(s);
        }
    }

    let sorted = crate::utils::safe_sort(scores);
    if sorted.is_empty() { anyhow::bail!("no valid scores for calibration"); }
    let q = crate::utils::conformal_quantile(&sorted, cfg.alpha);

    let per_label_q = if cfg.mondrian {
        let mut m = HashMap::new();
        for (li, v) in per_label.into_iter() {
            let vv = crate::utils::safe_sort(v);
            if vv.is_empty() { continue; }
            let qq = crate::utils::conformal_quantile(&vv, cfg.alpha);
            let key = if let Some(names) = &canon_labels { names[li].clone() } else { format!("{}", li) };
            m.insert(key, qq);
        }
        Some(m)
    } else { None };

    Ok(CalibModel {
        task: "class".into(),
        alpha: cfg.alpha,
        global_q: q,
        per_label_q,
        labels: canon_labels,
        n: sorted.len(),
    })
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Deserialize)]
struct RegrCalibRowOnnx { x: Vec<f32>, y_true: f64 }

#[cfg(feature = "onnx")]
/// Fits a regression calibration model by running an ONNX model to obtain
/// `y_pred` values for each feature vector.
fn fit_regr_from_jsonl_onnx(path: &str, cfg: CalibConfig, onnx: crate::onnx::OnnxOptions) -> Result<CalibModel> {
    use crate::onnx::OnnxRunner;
    let reader = BufReader::new(File::open(path).with_context(|| "open regr calib file (onnx)")?);
    let rows: Vec<RegrCalibRowOnnx> = jsonl_deser(reader, cfg.max_rows)?;
    if rows.is_empty() { anyhow::bail!("no calibration rows"); }

    let mut runner = OnnxRunner::new(&onnx)?;
    let mut scores: Vec<f64> = Vec::with_capacity(rows.len());
    for r in rows.into_iter() {
        let out = runner.infer_vec_f32(&r.x)?;
        if out.is_empty() { anyhow::bail!("onnx model produced empty output"); }
        let y_pred = out[0] as f64;
        let s = regr_score(r.y_true, y_pred);
        if s.is_finite() { scores.push(s); }
    }
    let sorted = safe_sort(scores);
    if sorted.is_empty() { anyhow::bail!("no valid scores for calibration"); }
    let q = conformal_quantile(&sorted, cfg.alpha);

    Ok(CalibModel {
        task: "regr".into(),
        alpha: cfg.alpha,
        global_q: q,
        per_label_q: None,
        labels: None,
        n: sorted.len(),
    })
}
