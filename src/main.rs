use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::fs::File;
use std::io::{BufReader, BufWriter};

use onnxconformal_rs::calibrator::{CalibConfig, CalibFileKind, CalibModel, CalibSummary};
use onnxconformal_rs::predictor::{self, PredConfig};

#[derive(Debug, Clone, ValueEnum)]
enum Task {
    Class,
    Regr,
}

#[derive(Parser, Debug)]
#[command(author, version, about="Conformal prediction in Rust")]
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
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Calibrate { task, alpha, input, output, mondrian, max_rows } => {
            let task_kind = match task { Task::Class => CalibFileKind::Classification, Task::Regr => CalibFileKind::Regression };
            let cfg = CalibConfig { alpha, mondrian, max_rows };
            let calib = CalibModel::fit_from_file(&input, task_kind, cfg)
                .with_context(|| "failed to fit calibration")?;
            calib.save(&output)?;
            let summary = CalibSummary::from(&calib);
            eprintln!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::Predict { task, calib, input, output, max_set_size, include_probs, max_rows } => {
            let calib = CalibModel::load(&calib).with_context(|| "failed to load calib json")?;
            let pred_cfg = PredConfig { max_set_size, include_probs, max_rows };
            let reader = BufReader::new(File::open(&input).with_context(|| "open input JSONL")?);
            let writer = BufWriter::new(File::create(&output).with_context(|| "create output JSONL")?);
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
