use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct TextOptions {
    pub tokenizer: String,
    pub max_len: Option<usize>,
    pub truncation: bool,
    pub padding: bool,
}

/// Thin wrapper over `tokenizers::Tokenizer` with simple truncation/padding.
pub struct TextTokenizer {
    tk: tokenizers::Tokenizer,
    max_len: Option<usize>,
    truncation: bool,
    padding: bool,
}

impl TextTokenizer {
    pub fn new(opts: &TextOptions) -> Result<Self> {
        let tk = tokenizers::Tokenizer::from_file(&opts.tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
            .with_context(|| format!("tokenizer file: {}", &opts.tokenizer))?;
        Ok(Self { tk, max_len: opts.max_len, truncation: opts.truncation, padding: opts.padding })
    }

    /// Encodes raw text into token IDs (i64). If `max_len` is set, applies
    /// manual truncation; if `padding` is true and `max_len` is set, pads with 0s.
    pub fn encode_ids_i64(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .tk
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        if let Some(m) = self.max_len {
            if self.truncation && ids.len() > m { ids.truncate(m); }
            if self.padding && ids.len() < m { ids.resize(m, 0i64); }
        }
        Ok(ids)
    }
}

