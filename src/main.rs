mod classifier;
mod llm;

use std::io::{self, BufRead};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::json;

use classifier::{Classifier, TrainOptions};
use llm::{LlmClient, LlmConfig, LlmVerdict};

#[derive(Parser, Debug)]
#[command(
    name = "rustfasttext",
    version,
    about = "fastText 小模型 + 大模型(LLM) 混合文本分类命令行工具"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// 训练一个 fastText 有监督分类小模型
    Train(TrainCmd),
    /// 只用小模型做预测
    Predict(PredictCmd),
    /// 小模型优先、大模型兜底的混合分类
    Analyze(AnalyzeCmd),
    /// 直接调用大模型（OpenAI 兼容 chat/completions 接口）
    Llm(LlmCmd),
    /// 查看小模型信息
    Info(InfoCmd),
}

#[derive(Args, Debug)]
struct TrainCmd {
    /// 训练数据，每行格式：`__label__xxx 文本内容`
    #[arg(short, long, default_value = "data/train.txt")]
    input: PathBuf,

    /// 模型输出路径
    #[arg(short, long, default_value = "model.bin")]
    output: PathBuf,

    #[arg(long, default_value_t = 25)]
    epoch: i32,

    #[arg(long, default_value_t = 0.5)]
    lr: f64,

    #[arg(long, default_value_t = 100)]
    dim: i32,

    #[arg(long, default_value_t = 2)]
    word_ngrams: i32,

    #[arg(long, default_value_t = 1)]
    min_count: i32,

    #[arg(long, default_value_t = 0)]
    minn: i32,

    #[arg(long, default_value_t = 0)]
    maxn: i32,

    /// 训练线程数，0 表示自动
    #[arg(long, default_value_t = 0)]
    threads: i32,

    #[arg(long, default_value = "__label__")]
    label_prefix: String,
}

#[derive(Args, Debug)]
struct PredictCmd {
    /// 小模型路径
    #[arg(short, long, default_value = "model.bin")]
    model: PathBuf,

    /// 待预测文本；不传则读取 --input 或 stdin
    #[arg(long)]
    text: Option<String>,

    /// 每行一条文本的文件
    #[arg(long)]
    input: Option<PathBuf>,

    /// 返回前 k 个标签
    #[arg(short = 'k', long, default_value_t = 3)]
    top_k: usize,

    #[arg(long, default_value_t = 0.0)]
    threshold: f32,

    /// 以 JSON 输出
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct AnalyzeCmd {
    #[arg(short, long, default_value = "model.bin")]
    model: PathBuf,

    #[arg(long)]
    text: Option<String>,

    #[arg(long)]
    input: Option<PathBuf>,

    #[arg(short = 'k', long, default_value_t = 3)]
    top_k: usize,

    /// 置信度阈值：小模型 top1 低于该值时调用大模型兜底
    #[arg(long, default_value_t = 0.6)]
    threshold: f32,

    #[arg(long)]
    json: bool,

    #[command(flatten)]
    llm: LlmOpts,
}

#[derive(Args, Debug)]
struct LlmCmd {
    #[arg(long)]
    text: Option<String>,

    #[arg(long)]
    input: Option<PathBuf>,

    /// system prompt
    #[arg(long, default_value = "你是一个有帮助的助手，请用简洁的中文回答。")]
    system: String,

    #[arg(long)]
    json: bool,

    #[command(flatten)]
    llm: LlmOpts,
}

#[derive(Args, Debug)]
struct LlmOpts {
    /// OpenAI 兼容服务地址，如 https://api.openai.com/v1
    #[arg(long, env = "LLM_BASE_URL", default_value = "")]
    base_url: String,

    #[arg(long, env = "LLM_API_KEY", default_value = "")]
    api_key: String,

    #[arg(long, env = "LLM_MODEL", default_value = "")]
    model: String,

    /// 超时时间（秒）
    #[arg(long, env = "LLM_TIMEOUT_SECS", default_value_t = 30)]
    timeout: u64,
}

#[derive(Args, Debug)]
struct InfoCmd {
    #[arg(short, long, default_value = "model.bin")]
    model: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(c) => cmd_train(c),
        Command::Predict(c) => cmd_predict(c),
        Command::Analyze(c) => cmd_analyze(c),
        Command::Llm(c) => cmd_llm(c),
        Command::Info(c) => cmd_info(c),
    }
}

fn cmd_train(c: TrainCmd) -> Result<()> {
    let opts = TrainOptions {
        input: c.input,
        output: c.output,
        epoch: c.epoch,
        lr: c.lr,
        dim: c.dim,
        word_ngrams: c.word_ngrams,
        min_count: c.min_count,
        minn: c.minn,
        maxn: c.maxn,
        threads: c.threads,
        label_prefix: c.label_prefix,
    };

    let labels_path = opts.input.clone();
    let output = opts.output.clone();
    let model = classifier::train(&opts)?;
    let labels = model.get_labels().0;

    println!("训练数据: {}", labels_path.display());
    println!("模型已保存: {}", output.display());
    println!("标签数量: {}", labels.len());
    println!("标签列表: {}", labels.join(", "));
    Ok(())
}

fn cmd_predict(c: PredictCmd) -> Result<()> {
    let clf = Classifier::load(&c.model)?;
    for text in read_texts(c.text, c.input)? {
        let preds = clf.predict(&text, c.top_k, c.threshold);
        if c.json {
            println!("{}", json!({
                "text": text.as_str(),
                "predictions": preds.iter().map(|p| json!({"label": p.label.as_str(), "prob": p.prob})).collect::<Vec<_>>(),
            }));
        } else if preds.is_empty() {
            println!("{}\t(no prediction)", text);
        } else {
            let hits: Vec<String> = preds
                .iter()
                .map(|p| format!("{}:{:.4}", p.label, p.prob))
                .collect();
            println!("{}\t{}", text, hits.join("  "));
        }
    }
    Ok(())
}

fn cmd_analyze(c: AnalyzeCmd) -> Result<()> {
    let clf = Classifier::load(&c.model)?;
    let labels = clf.labels();
    let cfg = LlmConfig::new(c.llm.base_url, c.llm.api_key, c.llm.model, c.llm.timeout);
    let client = if cfg.is_configured() {
        Some(LlmClient::new(&cfg)?)
    } else {
        None
    };

    for text in read_texts(c.text, c.input)? {
        let preds = clf.predict(&text, c.top_k, 0.0);
        let top = preds.first();
        let confident = top.map(|p| p.prob >= c.threshold).unwrap_or(false);

        let mut out = json!({
            "text": text.as_str(),
            "threshold": c.threshold,
            "small_model": preds.iter().map(|p| json!({"label": p.label.as_str(), "prob": p.prob})).collect::<Vec<_>>(),
        });

        let verdict: Option<LlmVerdict> = if confident {
            None
        } else {
            client.as_ref().and_then(|cli| match cli.classify(&text, &labels) {
                Ok(v) => Some(v),
                Err(e) => {
                    out["llm_error"] = json!(e.to_string());
                    None
                }
            })
        };

        match verdict {
            None => {
                out["source"] = json!("fasttext");
                out["label"] = json!(top.map(|p| p.label.as_str()).unwrap_or(""));
                out["prob"] = json!(top.map(|p| p.prob).unwrap_or(0.0));
                if !confident {
                    out["note"] = json!("未配置大模型（设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后自动启用兜底）");
                }
            }
            Some(v) => {
                out["source"] = json!("llm");
                out["label"] = json!(v.label.as_str());
                out["reason"] = json!(v.reason.as_str());
                out["prob"] = json!(top.map(|p| p.prob).unwrap_or(0.0));
            }
        }

        if c.json {
            println!("{}", out);
        } else {
            println!(
                "{}\t[{}]\treason={}",
                out["label"].as_str().unwrap_or(""),
                out["source"].as_str().unwrap_or(""),
                out["reason"].as_str().unwrap_or("-")
            );
        }
    }

    Ok(())
}

fn cmd_llm(c: LlmCmd) -> Result<()> {
    let cfg = LlmConfig::new(c.llm.base_url, c.llm.api_key, c.llm.model, c.llm.timeout);
    if !cfg.is_configured() {
        anyhow::bail!("未配置大模型：请通过 --base-url / --api-key / --model 或 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量提供");
    }
    let client = LlmClient::new(&cfg)?;
    for text in read_texts(c.text, c.input)? {
        let answer = client.chat(&c.system, &text)?;
        if c.json {
            println!("{}", json!({ "text": text.as_str(), "answer": answer.as_str() }));
        } else {
            println!("{}\t{}", text, answer);
        }
    }
    Ok(())
}

fn cmd_info(c: InfoCmd) -> Result<()> {
    let clf = Classifier::load(&c.model)?;
    let labels = clf.labels();
    println!("模型文件: {}", c.model.display());
    println!("向量维度: {}", clf.dim());
    println!("词表大小: {}", clf.vocab_size());
    println!("标签数量: {}", labels.len());
    println!("是否量化: {}", clf.is_quantized());
    println!("标签列表: {}", labels.join(", "));
    Ok(())
}

/// 文本来源优先级：--text > --input > stdin
fn read_texts(text: Option<String>, input: Option<PathBuf>) -> Result<Vec<String>> {
    if let Some(t) = text {
        return Ok(vec![t]);
    }
    if let Some(p) = input {
        let content =
            std::fs::read_to_string(&p).with_context(|| format!("读取文件失败: {}", p.display()))?;
        return Ok(non_empty_lines(&content));
    }

    let stdin = io::stdin();
    let mut out = Vec::new();
    for line in stdin.lock().lines() {
        let line = line.context("读取标准输入失败")?;
        if !line.trim().is_empty() {
            out.push(line.trim().to_string());
        }
    }
    Ok(out)
}

fn non_empty_lines(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}
