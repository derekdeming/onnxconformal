//! Minimal ONNX Runtime runner used by the CLI.
//!
//! Provides a small, single‑input/single‑output abstraction over the `ort`
//! crate, so most of the codebase avoids direct ORT APIs. Input/output tensor
//! element types are detected and handled (e.g., `f32`, `bool`).

use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use std::path::Path;

/// Executes a single‑input, single‑output ONNX model for batch size 1.
pub struct OnnxRunner {
    session: Session,
    input_name: String,
    output_name: String,
    in_dtype: TensorElementType,
    out_dtype: TensorElementType,
}

#[derive(Debug, Clone)]
/// Loader options for creating an [`OnnxRunner`].
pub struct OnnxOptions {
    pub model: String,
    pub input_name: Option<String>,
    pub output_name: Option<String>,
}

impl OnnxRunner {
    /// Loads an ONNX model and prepares a session.
    ///
    /// Uses the first input/output by default unless names are provided, and
    /// captures input/output tensor element types for adaptive conversion.
    pub fn new(opts: &OnnxOptions) -> Result<Self> {
        if !Path::new(&opts.model).exists() {
            bail!("ONNX model not found: {}", opts.model);
        }
        let session = ort::session::Session::builder()
            .context("failed to build ORT session builder")?
            .commit_from_file(&opts.model)
            .with_context(|| format!("failed to load model: {}", &opts.model))?;

        let default_in = session
            .inputs
            .get(0)
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow::anyhow!("model has no inputs"))?;
        let default_out = session
            .outputs
            .get(0)
            .map(|o| o.name.clone())
            .ok_or_else(|| anyhow::anyhow!("model has no outputs"))?;
        let in_dtype = match &session.inputs.get(0).unwrap().input_type {
            ValueType::Tensor { ty, .. } => *ty,
            _ => anyhow::bail!("only tensor inputs supported"),
        };
        let out_dtype = match &session.outputs.get(0).unwrap().output_type {
            ValueType::Tensor { ty, .. } => *ty,
            _ => anyhow::bail!("only tensor outputs supported"),
        };

        let input_name = opts.input_name.clone().unwrap_or(default_in);
        let output_name = opts.output_name.clone().unwrap_or(default_out);

        Ok(Self { session, input_name, output_name, in_dtype, out_dtype })
    }

    /// Runs inference for a single 1‑D feature vector (batch size 1) of f32.
    /// Returns a flattened `Vec<f32>`; boolean/int outputs are converted.
    pub fn infer_vec_f32(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let inputs_val = match self.in_dtype {
            TensorElementType::Float32 => {
                let tensor = ort::value::Tensor::<f32>::from_array(([1usize, x.len()], x.to_vec().into_boxed_slice()))?;
                ort::inputs![ self.input_name.as_str() => tensor ]
            }
            TensorElementType::Bool => {
                let v: Vec<bool> = x.iter().map(|&f| f > 0.0).collect();
                let tensor = ort::value::Tensor::<bool>::from_array(([1usize, v.len()], v.into_boxed_slice()))?;
                ort::inputs![ self.input_name.as_str() => tensor ]
            }
            other => anyhow::bail!("unsupported input dtype for f32 input: {:?}", other),
        };

        let outputs = self.session
            .run(inputs_val)
            .context("onnx run failed")?;

        let value = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("output '{}' missing from model run", self.output_name))?;
        let vec_f32: Vec<f32> = match self.out_dtype {
            TensorElementType::Float32 => {
                let tref: ort::value::TensorRef<f32> = value.downcast_ref().context("output not f32 tensor")?;
                let (_, data) = tref.extract_tensor();
                data.to_vec()
            }
            TensorElementType::Bool => {
                let tref: ort::value::TensorRef<bool> = value.downcast_ref().context("output not bool tensor")?;
                let (_, data) = tref.extract_tensor();
                data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
            }
            TensorElementType::Int64 => {
                let tref: ort::value::TensorRef<i64> = value.downcast_ref().context("output not i64 tensor")?;
                let (_, data) = tref.extract_tensor();
                data.iter().map(|&v| v as f32).collect()
            }
            TensorElementType::Int32 => {
                let tref: ort::value::TensorRef<i32> = value.downcast_ref().context("output not i32 tensor")?;
                let (_, data) = tref.extract_tensor();
                data.iter().map(|&v| v as f32).collect()
            }
            other => anyhow::bail!("unsupported output dtype: {:?}", other),
        };
        Ok(vec_f32)
    }

    /// Runs inference for a single 1‑D feature vector (batch size 1) of i64 IDs.
    /// Returns a flattened `Vec<f32>`; boolean/int outputs are converted.
    pub fn infer_vec_i64(&mut self, x: &[i64]) -> Result<Vec<f32>> {
        let inputs_val = match self.in_dtype {
            TensorElementType::Int64 => {
                let tensor = ort::value::Tensor::<i64>::from_array(([1usize, x.len()], x.to_vec().into_boxed_slice()))?;
                ort::inputs![ self.input_name.as_str() => tensor ]
            }
            TensorElementType::Int32 => {
                let v: Vec<i32> = x.iter().map(|&v| v as i32).collect();
                let tensor = ort::value::Tensor::<i32>::from_array(([1usize, v.len()], v.into_boxed_slice()))?;
                ort::inputs![ self.input_name.as_str() => tensor ]
            }
            other => anyhow::bail!("unsupported input dtype for i64 input: {:?}", other),
        };

        let outputs = self.session
            .run(inputs_val)
            .context("onnx run failed")?;

        let value = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("output '{}' missing from model run", self.output_name))?;
        let vec_f32: Vec<f32> = match self.out_dtype {
            TensorElementType::Float32 => {
                let tref: ort::value::TensorRef<f32> = value.downcast_ref().context("output not f32 tensor")?;
                let (_, data) = tref.extract_tensor();
                data.to_vec()
            }
            TensorElementType::Bool => {
                let tref: ort::value::TensorRef<bool> = value.downcast_ref().context("output not bool tensor")?;
                let (_, data) = tref.extract_tensor();
                data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
            }
            TensorElementType::Int64 => {
                let tref: ort::value::TensorRef<i64> = value.downcast_ref().context("output not i64 tensor")?;
                let (_, data) = tref.extract_tensor();
                data.iter().map(|&v| v as f32).collect()
            }
            TensorElementType::Int32 => {
                let tref: ort::value::TensorRef<i32> = value.downcast_ref().context("output not i32 tensor")?;
                let (_, data) = tref.extract_tensor();
                data.iter().map(|&v| v as f32).collect()
            }
            other => anyhow::bail!("unsupported output dtype: {:?}", other),
        };
        Ok(vec_f32)
    }
}
