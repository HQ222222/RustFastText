# RustFastText

Rust + [fasttext-rs](https://github.com/messense/fasttext-rs) 小模型，配合大模型（LLM）兜底的文本分类命令行工具。

- **小模型**：`fasttext = "0.8"`（**纯 Rust 实现**，不需要 C++ 工具链 / 不需要 Docker），毫秒级分类。
- **大模型**：OpenAI 兼容 `chat/completions` 接口，仅在**小模型置信度不足**时调用，兼顾成本与效果。
- **开箱即用**：GitHub Actions 直接产出 `aarch64-unknown-linux-gnu` 与 Windows x86_64 两个**单文件二进制**，拷贝即可运行，服务器上无需 Rust / Python / 任何依赖。

## 快速开始

```bash
# 1. 训练小模型（data/train.txt 每行格式: __label__xxx 文本内容）
rustfasttext train --input data/train.txt --output model.bin --epoch 25

# 2. 小模型预测
rustfasttext predict --model model.bin --text "mysql connection pool is exhausted" -k 2
# => __label__database:0.9213  __label__hardware:0.0512

# 3. 小模型优先 + 大模型兜底（top1 置信度 < 0.6 时调用大模型）
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-xxx
export LLM_MODEL=gpt-4o-mini
rustfasttext analyze --model model.bin --text "the raid controller beeps and the disk is gone" --json

# 4. 直接对话式调用大模型
rustfasttext llm --text "用一句话解释什么是 fastText"

# 5. 查看模型信息
rustfasttext info --model model.bin
```

文本来源优先级：`--text` > `--input 文件` > `stdin`（每行一条）。

### 大模型配置

| 环境变量 | CLI 参数 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | `--llm-base-url` | OpenAI 兼容地址，如 `https://api.openai.com/v1`、`https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_API_KEY` | `--llm-api-key` | API Key |
| `LLM_MODEL` | `--llm-model` | 模型名 |
| `LLM_TIMEOUT_SECS` | `--llm-timeout` | 超时秒数，默认 30 |

未配置大模型时，`analyze` 自动降级为纯小模型结果（不会报错，输出中 `source=fasttext` 并带 `note` 提示）。

## GitHub Actions 流水线

`.github/workflows/build.yml` 在 `push(main)` / `PR` / `tag(v*)` / 手动触发时运行：

| Job | Runner | 方式 | 产物 |
| --- | --- | --- | --- |
| `build-aarch64` | `ubuntu-22.04-arm` | **ARM64 原生编译**（无需 Docker / 交叉工具链） | `rustfasttext`（aarch64-unknown-linux-gnu，ELF ARM aarch64） |
| `build-windows` | `windows-latest` | MSVC 原生编译 | `rustfasttext.exe`（x86_64-pc-windows-msvc） |
| `release` | `ubuntu-latest` | 打 `v*` tag 时执行 | 自动创建 GitHub Release 并附带两个二进制 |

每个平台都会做真实运行验证：

- aarch64：`uname -m` + `file` 校验为 ARM aarch64 可执行文件，然后在 ARM runner 上原生执行
  `--version` / `train` / `info` / `predict` / `analyze`，并断言输出包含 `__label__`。
- Windows：原生执行 `--version` / `train` / `predict` / `info` / `analyze`，同样断言输出包含 `__label__`。

本地**不需要**安装任何交叉编译环境，也不需要在本地编译，直接推送即可：

```bash
git add .
git commit -m "feat: xxx"
git tag v0.1.0        # 可选，打 tag 会额外发布 Release
git push origin main --tags
```

在 Actions 页面下载 `rustfasttext-aarch64-unknown-linux-gnu` 或 `rustfasttext-x86_64-pc-windows-msvc` 即可。

### aarch64 服务器部署

```bash
# 服务器上只需要这一个文件
scp rustfasttext user@arm-server:/usr/local/bin/
ssh user@arm-server "chmod +x /usr/local/bin/rustfasttext && rustfasttext --version"
```

- 产物在 `ubuntu-22.04-arm` 上原生编译，依赖 glibc ≥ 2.35（Ubuntu 22.04+/Debian 12+ 均满足）。
- 无 OpenSSL / CA 证书依赖：TLS 使用 `rustls` + 内置 `webpki-roots` 根证书。
- 单文件静态分发，放到哪都能跑，不需要 Python / Rust / 模型服务框架。

## 本地开发（可选）

```bash
cargo build --release
cargo run --release -- train --input data/train.txt --output model.bin
cargo run --release -- predict --model model.bin --text "vpn tunnel keeps dropping"
```

## 目录结构

```
src/main.rs                 CLI 入口（train / predict / analyze / llm / info）
src/classifier.rs           fastText 小模型封装
src/llm.rs                  大模型（OpenAI 兼容接口）客户端与结果解析
data/train.txt              示例训练数据（5 类运维工单，125 条）
data/test.txt               示例测试数据
.github/workflows/build.yml  CI 流水线
```

## License

MIT
