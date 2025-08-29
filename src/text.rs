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
        Ok(Self {
            tk,
            max_len: opts.max_len,
            truncation: opts.truncation,
            padding: opts.padding,
        })
    }

    /// Encodes raw text into token IDs (i64). If `max_len` is set, applies
    /// manual truncation; if `padding` is true and `max_len` is set, pads with 0s.
    pub fn encode_ids_i64(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .tk
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        self.apply_pad_trunc(&mut ids);
        Ok(ids)
    }

    /// Encodes and returns (ids, attention_mask, type_ids) as i64 vectors,
    /// applying the same truncation/padding policy to all.
    pub fn encode_with_aux_i64(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let enc = self
            .tk
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        let mut mask: Vec<i64> = enc
            .get_attention_mask()
            .iter()
            .map(|&v| (v as i64))
            .collect();
        let mut type_ids: Vec<i64> = enc.get_type_ids().iter().map(|&v| (v as i64)).collect();
        self.apply_pad_trunc(&mut ids);
        // Ensure mask/type_ids match ids length after policy
        self.apply_pad_trunc(&mut mask);
        self.apply_pad_trunc(&mut type_ids);
        Ok((ids, mask, type_ids))
    }

    fn apply_pad_trunc(&self, v: &mut Vec<i64>) {
        if let Some(m) = self.max_len {
            if self.truncation && v.len() > m {
                v.truncate(m);
            }
            if self.padding && v.len() < m {
                v.resize(m, 0i64);
            }
        }
    }
}
