//! Minimal ONNX Runtime runner used by the CLI.
//!
//! Provides a small abstraction over the `ort` crate, supporting:
//! - Single or multi-input feeds (1–3) for text-style integer inputs
//! - Named output selection (primary or alternates)
//! - Common tensor dtypes: f32, bool, i64, i32

use anyhow::{bail, Context, Result};
use ort::session::{Session, SessionOutputs};
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
    output_names: Vec<String>,
}

#[derive(Debug, Clone)]
/// Loader options for creating an [`OnnxRunner`].
pub struct OnnxOptions {
    pub model: String,
    pub input_name: Option<String>,
    pub output_name: Option<String>,
    pub input_names: Option<Vec<String>>,
    pub output_names: Option<Vec<String>>,
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

        let input_name = opts.input_name.clone().unwrap_or(default_in.clone());
        let output_name = opts.output_name.clone().unwrap_or(default_out.clone());
        let output_names = opts.output_names.clone().unwrap_or_else(|| vec![default_out]);
        Ok(Self { session, input_name, output_name, in_dtype, out_dtype, output_names })
    }

    /// Runs inference for a single 1‑D feature vector (batch size 1) of f32.
    /// Returns a flattened `Vec<f32>`; boolean/int outputs are converted.
    pub fn infer_vec_f32(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        // Use dtype of selected input name; error if unsupported.
        let in_dtype = self
            .dtype_for_input(self.input_name.as_str())
            .unwrap_or(self.in_dtype);
        let inputs_val = match in_dtype {
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
        // Choose output name and dtype; if user requested a name, it must exist.
        let (out_name, out_dtype) = self.select_output_name_and_dtype()?;
        let outputs = self.session.run(inputs_val).context("onnx run failed")?;
        Self::extract_output_by_name(outputs, out_dtype, out_name.as_str())
    }

    /// Runs inference for a single 1‑D feature vector (batch size 1) of i64 IDs.
    /// Returns a flattened `Vec<f32>`; boolean/int outputs are converted.
    pub fn infer_vec_i64(&mut self, x: &[i64]) -> Result<Vec<f32>> {
        let in_dtype = self
            .dtype_for_input(self.input_name.as_str())
            .unwrap_or(self.in_dtype);
        let inputs_val = match in_dtype {
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
        let (out_name, out_dtype) = self.select_output_name_and_dtype()?;
        let outputs = self.session.run(inputs_val).context("onnx run failed")?;
        Self::extract_output_by_name(outputs, out_dtype, out_name.as_str())
    }

    fn select_output_name_and_dtype(&self) -> Result<(String, TensorElementType)> {
        // If explicit names are provided, they must resolve; otherwise default to first.
        if !self.output_name.is_empty() || !self.output_names.is_empty() {
            let mut names: Vec<&str> = Vec::new();
            if !self.output_name.is_empty() { names.push(self.output_name.as_str()); }
            for n in &self.output_names { names.push(n.as_str()); }
            for cand in names {
                if let Some(info) = self.session.outputs.iter().find(|o| o.name == cand) {
                    let ty = match &info.output_type { ValueType::Tensor { ty, .. } => *ty, _ => self.out_dtype };
                    return Ok((cand.to_string(), ty));
                }
            }
            anyhow::bail!("requested output not found; use --onnx-output/--onnx-outputs with a valid name");
        }
        let info = self.session.outputs.get(0).ok_or_else(|| anyhow::anyhow!("model has no outputs"))?;
        let ty = match &info.output_type { ValueType::Tensor { ty, .. } => *ty, _ => self.out_dtype };
        Ok((info.name.clone(), ty))
    }

    /// Multi-input (1–3) variant for integer (text-like) inputs. Each item is
    /// a pair of (input_name, data). Shapes are assumed `[1, len]`.
    pub fn infer_i64_named(&mut self, items: &[(&str, &[i64])]) -> Result<Vec<f32>> {
        use TensorElementType as T;
        match items.len() {
            0 => anyhow::bail!("no inputs provided"),
            1 => {
                let (n1, x1) = items[0];
                let dt1 = self.dtype_for_input(n1).unwrap_or(T::Int64);
                let inputs_val = match dt1 {
                    T::Int64 => {
                        let t1 = ort::value::Tensor::<i64>::from_array(([1usize, x1.len()], x1.to_vec().into_boxed_slice()))?;
                        ort::inputs![ n1 => t1 ]
                    }
                    T::Int32 => {
                        let v1: Vec<i32> = x1.iter().map(|&v| v as i32).collect();
                        let t1 = ort::value::Tensor::<i32>::from_array(([1usize, v1.len()], v1.into_boxed_slice()))?;
                        ort::inputs![ n1 => t1 ]
                    }
                    other => anyhow::bail!("unsupported input dtype for {}: {:?}", n1, other),
                };
                let (out_name, out_dtype) = self.select_output_name_and_dtype()?;
                let outputs = self.session.run(inputs_val).context("onnx run failed")?;
                Self::extract_output_by_name(outputs, out_dtype, out_name.as_str())
            }
            2 => {
                let (n1, x1) = items[0];
                let (n2, x2) = items[1];
                let d1 = self.dtype_for_input(n1).unwrap_or(T::Int64);
                let d2 = self.dtype_for_input(n2).unwrap_or(T::Int64);
                let inputs_val = match (d1, d2) {
                    (T::Int64, T::Int64) => {
                        let t1 = ort::value::Tensor::<i64>::from_array(([1usize, x1.len()], x1.to_vec().into_boxed_slice()))?;
                        let t2 = ort::value::Tensor::<i64>::from_array(([1usize, x2.len()], x2.to_vec().into_boxed_slice()))?;
                        ort::inputs![ n1 => t1, n2 => t2 ]
                    }
                    (T::Int32, T::Int64) => {
                        let v1: Vec<i32> = x1.iter().map(|&v| v as i32).collect();
                        let t1 = ort::value::Tensor::<i32>::from_array(([1usize, v1.len()], v1.into_boxed_slice()))?;
                        let t2 = ort::value::Tensor::<i64>::from_array(([1usize, x2.len()], x2.to_vec().into_boxed_slice()))?;
                        ort::inputs![ n1 => t1, n2 => t2 ]
                    }
                    (T::Int64, T::Int32) => {
                        let t1 = ort::value::Tensor::<i64>::from_array(([1usize, x1.len()], x1.to_vec().into_boxed_slice()))?;
                        let v2: Vec<i32> = x2.iter().map(|&v| v as i32).collect();
                        let t2 = ort::value::Tensor::<i32>::from_array(([1usize, v2.len()], v2.into_boxed_slice()))?;
                        ort::inputs![ n1 => t1, n2 => t2 ]
                    }
                    (T::Int32, T::Int32) => {
                        let v1: Vec<i32> = x1.iter().map(|&v| v as i32).collect();
                        let v2: Vec<i32> = x2.iter().map(|&v| v as i32).collect();
                        let t1 = ort::value::Tensor::<i32>::from_array(([1usize, v1.len()], v1.into_boxed_slice()))?;
                        let t2 = ort::value::Tensor::<i32>::from_array(([1usize, v2.len()], v2.into_boxed_slice()))?;
                        ort::inputs![ n1 => t1, n2 => t2 ]
                    }
                    other => anyhow::bail!("unsupported input dtype combination: {:?}", other),
                };
                let (out_name, out_dtype) = self.select_output_name_and_dtype()?;
                let outputs = self.session.run(inputs_val).context("onnx run failed")?;
                Self::extract_output_by_name(outputs, out_dtype, out_name.as_str())
            }
            3 => {
                let (n1, x1) = items[0];
                let (n2, x2) = items[1];
                let (n3, x3) = items[2];
                let d1 = self.dtype_for_input(n1).unwrap_or(T::Int64);
                let d2 = self.dtype_for_input(n2).unwrap_or(T::Int64);
                let d3 = self.dtype_for_input(n3).unwrap_or(T::Int64);
                let to_i32 = |xs: &[i64]| xs.iter().map(|&v| v as i32).collect::<Vec<i32>>();
                let t1_i64 = ort::value::Tensor::<i64>::from_array(([1usize, x1.len()], x1.to_vec().into_boxed_slice()))?;
                let t1_i32 = ort::value::Tensor::<i32>::from_array(([1usize, x1.len()], to_i32(x1).into_boxed_slice()))?;
                let t2_i64 = ort::value::Tensor::<i64>::from_array(([1usize, x2.len()], x2.to_vec().into_boxed_slice()))?;
                let t2_i32 = ort::value::Tensor::<i32>::from_array(([1usize, x2.len()], to_i32(x2).into_boxed_slice()))?;
                let t3_i64 = ort::value::Tensor::<i64>::from_array(([1usize, x3.len()], x3.to_vec().into_boxed_slice()))?;
                let t3_i32 = ort::value::Tensor::<i32>::from_array(([1usize, x3.len()], to_i32(x3).into_boxed_slice()))?;
                let inputs_val = match (d1, d2, d3) {
                    (T::Int64, T::Int64, T::Int64) => ort::inputs![ n1 => t1_i64, n2 => t2_i64, n3 => t3_i64 ],
                    (T::Int32, T::Int64, T::Int64) => ort::inputs![ n1 => t1_i32, n2 => t2_i64, n3 => t3_i64 ],
                    (T::Int64, T::Int32, T::Int64) => ort::inputs![ n1 => t1_i64, n2 => t2_i32, n3 => t3_i64 ],
                    (T::Int64, T::Int64, T::Int32) => ort::inputs![ n1 => t1_i64, n2 => t2_i64, n3 => t3_i32 ],
                    (T::Int32, T::Int32, T::Int64) => ort::inputs![ n1 => t1_i32, n2 => t2_i32, n3 => t3_i64 ],
                    (T::Int32, T::Int64, T::Int32) => ort::inputs![ n1 => t1_i32, n2 => t2_i64, n3 => t3_i32 ],
                    (T::Int64, T::Int32, T::Int32) => ort::inputs![ n1 => t1_i64, n2 => t2_i32, n3 => t3_i32 ],
                    (T::Int32, T::Int32, T::Int32) => ort::inputs![ n1 => t1_i32, n2 => t2_i32, n3 => t3_i32 ],
                    other => anyhow::bail!("unsupported input dtype combination: {:?}", other),
                };
                let (out_name, out_dtype) = self.select_output_name_and_dtype()?;
                let outputs = self.session.run(inputs_val).context("onnx run failed")?;
                Self::extract_output_by_name(outputs, out_dtype, out_name.as_str())
            }
            _ => anyhow::bail!("infer_i64_named supports up to 3 inputs"),
        }
    }

    fn dtype_for_input(&self, name: &str) -> Option<TensorElementType> {
        self.session.inputs.iter().find(|i| i.name == name).and_then(|i| match &i.input_type { ValueType::Tensor { ty, .. } => Some(*ty), _ => None })
    }

    fn extract_output_by_name(outputs: SessionOutputs, out_dtype: TensorElementType, output_name: &str) -> Result<Vec<f32>> {
        let value = outputs
            .get(output_name)
            .ok_or_else(|| anyhow::anyhow!("output '{}' missing from model run", output_name))?;
        let vec_f32: Vec<f32> = match out_dtype {
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
