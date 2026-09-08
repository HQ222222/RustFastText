use std::path::{Path, PathBuf};
use std::thread::available_parallelism;
use std::time::Instant;

use anyhow::{Context, Result};
use fasttext::args::{Args, ModelName};
use fasttext::{FastText, Prediction};

/// 训练小模型（fastText supervised）所需的参数。
pub struct TrainOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub epoch: i32,
    pub lr: f64,
    pub dim: i32,
    pub word_ngrams: i32,
    pub min_count: i32,
    pub minn: i32,
    pub maxn: i32,
    /// 0 表示自动取 CPU 核心数
    pub threads: i32,
    pub label_prefix: String,
}

/// 训练一个 fastText 有监督分类模型，并保存到 `opts.output`。
pub fn train(opts: &TrainOptions) -> Result<FastText> {
    if !opts.input.exists() {
        anyhow::bail!("训练数据不存在: {}", opts.input.display());
    }

    let mut args = Args::new();
    args.apply_supervised_defaults();
    args.model = ModelName::Supervised;
    args.input = opts.input.clone();
    args.output = opts.output.clone();
    args.label = opts.label_prefix.clone();
    args.epoch = opts.epoch;
    args.lr = opts.lr;
    args.dim = opts.dim;
    args.word_ngrams = opts.word_ngrams;
    args.min_count = opts.min_count;
    args.minn = opts.minn;
    args.maxn = opts.maxn;
    // 使用 word n-gram 时必须保留 bucket
    args.bucket = 2_000_000;
    args.thread = if opts.threads > 0 {
        opts.threads
    } else {
        available_parallelism().map(|n| n.get() as i32).unwrap_or(4)
    };
    args.verbose = 1;

    let started = Instant::now();
    let model = FastText::train(args).context("fastText 训练失败")?;
    eprintln!(
        "[fasttext] 训练完成: epoch={} dim={} 耗时={:?}",
        opts.epoch,
        opts.dim,
        started.elapsed()
    );

    if let Some(parent) = opts.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("创建目录失败: {}", parent.display()))?;
        }
    }
    model
        .save_model(&opts.output)
        .with_context(|| format!("保存模型失败: {}", opts.output.display()))?;

    Ok(model)
}

/// 小模型封装：负责加载与预测。
pub struct Classifier {
    model: FastText,
}

impl Classifier {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let model = FastText::load_model(path)
            .with_context(|| format!("加载模型失败: {}", path.display()))?;
        Ok(Self { model })
    }

    pub fn predict(&self, text: &str, k: usize, threshold: f32) -> Vec<Prediction> {
        self.model.predict(text, k, threshold)
    }

    /// 模型内部的标签列表（含 `__label__` 前缀）
    pub fn labels(&self) -> Vec<String> {
        self.model.get_labels().0
    }

    pub fn vocab_size(&self) -> usize {
        self.model.get_vocab().0.len()
    }

    pub fn dim(&self) -> i32 {
        self.model.get_dimension()
    }

    pub fn is_quantized(&self) -> bool {
        self.model.is_quant()
    }
}
