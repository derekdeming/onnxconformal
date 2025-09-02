use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel, CalibSummary};
use onnxconformal_rs::predictor::{self, PredConfig};

#[derive(Debug, Clone, ValueEnum)]
enum Task {
    Class,
    Regr,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Conformal prediction in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Fit calibration thresholds from a holdout file (JSONL)
    Calibrate {
        /// Task type: class|regr
        #[arg(long, value_enum)]
        task: Task,
        /// Miscoverage alpha (e.g., 0.1 for 90% coverage)
        #[arg(long, default_value_t = 0.1)]
        alpha: f64,
        /// Input calibration JSONL path
        #[arg(long)]
        input: String,
        /// Output calib JSON path
        #[arg(long)]
        output: String,
        /// Use per-class (Mondrian) calibration for classification
        #[arg(long, default_value_t = false)]
        mondrian: bool,
        /// Maximum rows to read (for quick experiments)
        #[arg(long)]
        max_rows: Option<usize>,
        /// Optional: ONNX model to run for scores (requires --features onnx)
        #[cfg(feature = "onnx")]
        #[arg(long)]
        onnx_model: Option<String>,
        /// Optional: ONNX input tensor name
        #[cfg(feature = "onnx")]
        #[arg(long)]
        onnx_input: Option<String>,
        /// Optional: multiple ONNX input names (comma-separated)
        #[cfg(feature = "onnx")]
        #[arg(long = "onnx-inputs", value_delimiter = ',')]
        onnx_inputs: Option<Vec<String>>,
        /// Optional: ONNX output tensor name
        #[cfg(feature = "onnx")]
        #[arg(long)]
        onnx_output: Option<String>,
        /// Optional: multiple ONNX output names (comma-separated)
        #[cfg(feature = "onnx")]
        #[arg(long = "onnx-outputs", value_delimiter = ',')]
        onnx_outputs: Option<Vec<String>>,
        /// Optional: text tokenization (requires --features onnx,text)
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long, value_name = "tokenizer.json")]
        tokenizer: Option<String>,
        /// Max token length for truncation/padding
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long)]
        max_len: Option<usize>,
        /// Enable truncation when sequence exceeds max_len
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long, default_value_t = false)]
        truncation: bool,
        /// Enable padding up to max_len
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long, default_value_t = false)]
        padding: bool,
    },
    /// Apply prediction sets/intervals to new data (JSONL)
    Predict {
        /// Task type: class|regr
        #[arg(long, value_enum)]
        task: Task,
        /// Calib JSON path
        #[arg(long)]
        calib: String,
        /// Input JSONL path (scores or y_hat)
        #[arg(long)]
        input: String,
        /// Output JSONL path
        #[arg(long)]
        output: String,
        /// For classification: cap size of prediction set
        #[arg(long)]
        max_set_size: Option<usize>,
        /// For classification: include set probabilities in output
        #[arg(long, default_value_t = false)]
        include_probs: bool,
        /// Maximum rows to read (for quick experiments)
        #[arg(long)]
        max_rows: Option<usize>,
        /// Optional: ONNX model to run for scores (requires --features onnx)
        #[cfg(feature = "onnx")]
        #[arg(long)]
        onnx_model: Option<String>,
        /// Optional: ONNX input tensor name
        #[cfg(feature = "onnx")]
        #[arg(long)]
        onnx_input: Option<String>,
        /// Optional: multiple ONNX input names (comma-separated)
        #[cfg(feature = "onnx")]
        #[arg(long = "onnx-inputs", value_delimiter = ',')]
        onnx_inputs: Option<Vec<String>>,
        /// Optional: ONNX output tensor name
        #[cfg(feature = "onnx")]
        #[arg(long)]
        onnx_output: Option<String>,
        /// Optional: multiple ONNX output names (comma-separated)
        #[cfg(feature = "onnx")]
        #[arg(long = "onnx-outputs", value_delimiter = ',')]
        onnx_outputs: Option<Vec<String>>,
        /// Optional: text tokenization (requires --features onnx,text)
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long, value_name = "tokenizer.json")]
        tokenizer: Option<String>,
        /// Max token length for truncation/padding
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long)]
        max_len: Option<usize>,
        /// Enable truncation when sequence exceeds max_len
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long, default_value_t = false)]
        truncation: bool,
        /// Enable padding up to max_len
        #[cfg(all(feature = "onnx", feature = "text"))]
        #[arg(long, default_value_t = false)]
        padding: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Calibrate {
            task,
            alpha,
            input,
            output,
            mondrian,
            max_rows,
            #[cfg(feature = "onnx")]
            onnx_model,
            #[cfg(feature = "onnx")]
            onnx_input,
            #[cfg(feature = "onnx")]
            onnx_inputs,
            #[cfg(feature = "onnx")]
            onnx_output,
            #[cfg(feature = "onnx")]
            onnx_outputs,
            #[cfg(all(feature = "onnx", feature = "text"))]
            tokenizer,
            #[cfg(all(feature = "onnx", feature = "text"))]
            max_len,
            #[cfg(all(feature = "onnx", feature = "text"))]
            truncation,
            #[cfg(all(feature = "onnx", feature = "text"))]
            padding,
        } => {
            let task_kind = match task {
                Task::Class => CalibFileKind::Classification,
                Task::Regr => CalibFileKind::Regression,
            };
            #[cfg(feature = "onnx")]
            let cfg = {
                let onnx = onnx_model
                    .as_ref()
                    .map(|m| onnxconformal_rs::onnx::OnnxOptions {
                        model: m.clone(),
                        input_name: onnx_input.clone(),
                        output_name: onnx_output.clone(),
                        input_names: onnx_inputs.clone(),
                        output_names: onnx_outputs.clone(),
                    });
                #[cfg(feature = "text")]
                let text = tokenizer
                    .as_ref()
                    .map(|tok| onnxconformal_rs::text::TextOptions {
                        tokenizer: tok.clone(),
                        max_len,
                        truncation,
                        padding,
                    });
                CalibConfig {
                    alpha,
                    mondrian,
                    max_rows,
                    onnx,
                    #[cfg(feature = "text")]
                    text,
                }
            };
            #[cfg(not(feature = "onnx"))]
            let cfg = CalibConfig {
                alpha,
                mondrian,
                max_rows,
            };
            let calib = CalibModel::fit_from_file(&input, task_kind, cfg)
                .with_context(|| "failed to fit calibration")?;
            if output == "-" {
                serde_json::to_writer_pretty(std::io::stdout(), &calib)?;
                println!();
            } else {
                calib.save(&output)?;
            }
            let summary = CalibSummary::from(&calib);
            eprintln!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::Predict {
            task,
            calib,
            input,
            output,
            max_set_size,
            include_probs,
            max_rows,
            #[cfg(feature = "onnx")]
            onnx_model,
            #[cfg(feature = "onnx")]
            onnx_input,
            #[cfg(feature = "onnx")]
            onnx_inputs,
            #[cfg(feature = "onnx")]
            onnx_output,
            #[cfg(feature = "onnx")]
            onnx_outputs,
            #[cfg(all(feature = "onnx", feature = "text"))]
            tokenizer,
            #[cfg(all(feature = "onnx", feature = "text"))]
            max_len,
            #[cfg(all(feature = "onnx", feature = "text"))]
            truncation,
            #[cfg(all(feature = "onnx", feature = "text"))]
            padding,
        } => {
            let calib = CalibModel::load(&calib).with_context(|| "failed to load calib json")?;
            #[cfg(feature = "onnx")]
            let pred_cfg = {
                let onnx = onnx_model
                    .as_ref()
                    .map(|m| onnxconformal_rs::onnx::OnnxOptions {
                        model: m.clone(),
                        input_name: onnx_input.clone(),
                        output_name: onnx_output.clone(),
                        input_names: onnx_inputs.clone(),
                        output_names: onnx_outputs.clone(),
                    });
                #[cfg(feature = "text")]
                let text = tokenizer
                    .as_ref()
                    .map(|tok| onnxconformal_rs::text::TextOptions {
                        tokenizer: tok.clone(),
                        max_len,
                        truncation,
                        padding,
                    });
                PredConfig {
                    max_set_size,
                    include_probs,
                    max_rows,
                    onnx,
                    #[cfg(feature = "text")]
                    text,
                }
            };
            #[cfg(not(feature = "onnx"))]
            let pred_cfg = PredConfig {
                max_set_size,
                include_probs,
                max_rows,
            };
            // Support streaming via '-' for stdin/stdout
            let reader: Box<dyn BufRead> = if input == "-" {
                Box::new(BufReader::new(std::io::stdin()))
            } else {
                Box::new(BufReader::new(
                    File::open(&input).with_context(|| "open input JSONL")?,
                ))
            };
            let writer: Box<dyn Write> = if output == "-" {
                Box::new(BufWriter::new(std::io::stdout()))
            } else {
                Box::new(BufWriter::new(
                    File::create(&output).with_context(|| "create output JSONL")?,
                ))
            };
            match task {
                Task::Class => {
                    predictor::predict_classification(&calib, reader, writer, pred_cfg)?;
                }
                Task::Regr => {
                    predictor::predict_regression(&calib, reader, writer, pred_cfg)?;
                }
            }
        }
    }
    Ok(())
}
