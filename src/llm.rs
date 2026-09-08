use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::Client;
use serde_json::{json, Value};

/// 大模型（OpenAI 兼容 chat/completions 接口）配置。
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// 例如 `https://api.openai.com/v1`、`https://dashscope.aliyuncs.com/compatible-mode/v1`
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub timeout: Duration,
}

impl LlmConfig {
    pub fn new(base_url: String, api_key: String, model: String, timeout_secs: u64) -> Self {
        Self {
            base_url,
            api_key,
            model,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.base_url.is_empty() && !self.api_key.is_empty() && !self.model.is_empty()
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

/// 大模型分类结果。
#[derive(Debug, Clone)]
pub struct LlmVerdict {
    pub label: String,
    pub reason: String,
    pub raw: String,
}

pub struct LlmClient {
    http: Client,
    cfg: LlmConfig,
}

impl LlmClient {
    pub fn new(cfg: &LlmConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .context("初始化 HTTP 客户端失败")?;
        Ok(Self {
            http,
            cfg: cfg.clone(),
        })
    }

    /// 通用对话：返回 `choices[0].message.content`
    pub fn chat(&self, system: &str, user: &str) -> Result<String> {
        let url = self.cfg.chat_completions_url();
        let body = json!({
            "model": self.cfg.model,
            "temperature": 0.0,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user }
            ]
        });

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json");
        if !self.cfg.api_key.is_empty() {
            req = req.bearer_auth(&self.cfg.api_key);
        }

        let resp = req
            .json(&body)
            .send()
            .with_context(|| format!("调用大模型失败: {}", url))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            bail!(
                "大模型返回异常状态码 {}: {}",
                status,
                truncate(&text, 500)
            );
        }

        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("大模型响应不是合法 JSON: {}", truncate(&text, 200)))?;
        let content = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "大模型响应缺少 choices[0].message.content: {}",
                    truncate(&text, 300)
                )
            })?;

        Ok(content.trim().to_string())
    }

    /// 让大模型在给定标签集合中做单选分类。
    pub fn classify(&self, text: &str, labels: &[String]) -> Result<LlmVerdict> {
        if labels.is_empty() {
            bail!("小模型没有可用标签，无法交给大模型分类");
        }
        let system = "你是一个严格的文本分类引擎。只能从候选标签中选择最合适的一个，\
                      并且必须只输出一行 JSON：{\"label\": \"<候选标签之一>\", \"reason\": \"<不超过 30 字的中文理由>\"}，\
                      不要输出代码块、不要输出任何额外文字。";
        let user = format!(
            "候选标签：\n- {}\n\n待分类文本：\n\"\"\"\n{}\n\"\"\"\n\n请输出 JSON。",
            labels.join("\n- "),
            text
        );

        let raw = self.chat(system, &user)?;
        let (label, reason) = parse_verdict(&raw, labels)?;
        Ok(LlmVerdict { label, reason, raw })
    }
}

/// 从大模型输出中解析出标签与理由。
fn parse_verdict(raw: &str, labels: &[String]) -> Result<(String, String)> {
    let cleaned = raw.trim();
    let json_part = {
        let s = cleaned
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        match (s.find('{'), s.rfind('}')) {
            (Some(start), Some(end)) if start <= end => &s[start..=end],
            _ => s,
        }
    };

    let value: Value = serde_json::from_str(json_part).unwrap_or(Value::Null);
    let reason = value
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let candidate = value.get("label").and_then(|v| v.as_str());

    if let Some(c) = candidate {
        if let Some(label) = normalize_label(c, labels) {
            return Ok((label, reason));
        }
    }
    // 兜底：直接在回复文本里找出现过的候选标签
    if let Some(hit) = labels.iter().find(|l| cleaned.contains(l.as_str())) {
        return Ok((hit.clone(), reason));
    }

    bail!(
        "大模型输出无法解析为候选标签: {}",
        truncate(cleaned, 200)
    )
}

fn normalize_label(candidate: &str, labels: &[String]) -> Option<String> {
    let c = candidate.trim().trim_matches('"');
    if let Some(found) = labels.iter().find(|l| l.eq_ignore_ascii_case(c)) {
        return Some(found.clone());
    }
    let with_prefix = format!("__label__{}", c.trim_start_matches("__label__"));
    labels
        .iter()
        .find(|l| l.eq_ignore_ascii_case(&with_prefix))
        .cloned()
}

fn truncate(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.to_string();
    }
    let head: String = chars.into_iter().take(max_chars).collect();
    format!("{}...(已截断)", head)
}
