use anyhow::{bail, Context, Result};
use ort::session::Session;
use std::path::Path;

// Thin wrapper around the `ort` crate for single-input, single-output models.
// This keeps our main code free from direct ORT APIs and easy to mock.

pub struct OnnxRunner {
    session: Session,
    input_name: String,
    output_name: String,
}

#[derive(Debug, Clone)]
pub struct OnnxOptions {
    pub model: String,
    pub input_name: Option<String>,
    pub output_name: Option<String>,
}

impl OnnxRunner {
    pub fn new(opts: &OnnxOptions) -> Result<Self> {
        if !Path::new(&opts.model).exists() {
            bail!("ONNX model not found: {}", opts.model);
        }
        // Create environment and session with reasonable defaults.
        let session = ort::session::Session::builder()
            .context("failed to build ORT session builder")?
            .commit_from_file(&opts.model)
            .with_context(|| format!("failed to load model: {}", &opts.model))?;

        // By default, use the first input/output tensor names.
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

        let input_name = opts.input_name.clone().unwrap_or(default_in);
        let output_name = opts.output_name.clone().unwrap_or(default_out);

        Ok(Self { session, input_name, output_name })
    }

    // Run inference for a single 1D feature vector (batch size 1).
    pub fn infer_vec_f32(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        // Build input tensor [1, D]
        let tensor = ort::value::Tensor::<f32>::from_array(([1usize, x.len()], x.to_vec().into_boxed_slice()))?;

        let outputs = self.session
            .run(ort::inputs![ self.input_name.as_str() => tensor ])
            .context("onnx run failed")?;

        let value = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("output '{}' missing from model run", self.output_name))?;

        let tref: ort::value::TensorRef<f32> = value
            .downcast_ref()
            .context("output is not a float32 tensor")?;
        let (_, data) = tref.extract_tensor();
        Ok(data.to_vec())
    }
}
