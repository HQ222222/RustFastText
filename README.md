# RustFastText

Rust + [fasttext-rs](https://github.com/messense/fasttext-rs) 小模型，配合大模型（LLM）兜底的文本分类命令行工具。

- **小模型**：`fasttext = "0.8"`（纯 Rust 实现，不需要 C++ 工具链），毫秒级分类。
- **大模型**：OpenAI 兼容 `chat/completions` 接口，仅在**小模型置信度不足**时调用，兼顾成本与效果。
- **开箱即用**：GitHub Actions 直接产出 `aarch64-unknown-linux-gnu` 与 Windows x86_64 两个**单文件二进制**，拷贝即可运行，服务器上无需 Rust / Python / 任何依赖。

## 快速开始

```bash
# 1. 训练小模型（data/train.txt 每行格式: __label__xxx 文本内容）
rustfasttext train --input data/train.txt --output model.bin --epoch 25

# 2. 小模型预测
rustfasttext predict --model model.bin --text "mysql connection pool is exhausted" -k 2

# 3. 小模型优先 + 大模型兜底（置信度 < 0.6 时调用大模型）
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
| `LLM_BASE_URL` | `--base-url` | OpenAI 兼容地址，如 `https://api.openai.com/v1`、`https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_API_KEY` | `--api-key` | API Key |
| `LLM_MODEL` | `--model` | 模型名 |
| `LLM_TIMEOUT_SECS` | `--timeout` | 超时秒数，默认 30 |

未配置大模型时，`analyze` 会自动降级为纯小模型结果（不会报错）。

## GitHub Actions 流水线

`.github/workflows/build.yml` 在 `push(main)` / `PR` / `tag(v*)` / 手动触发时运行：

| Job | Runner | 方式 | 产物 |
| --- | --- | --- | --- |
| `build-aarch64` | `ubuntu-latest` | `cross`（Docker + aarch64 交叉工具链） | `rustfasttext`（aarch64-unknown-linux-gnu，ELF 64-bit ARM） |
| `build-windows` | `windows-latest` | 原生 MSVC | `rustfasttext.exe`（x86_64-pc-windows-msvc） |
| `release` | `ubuntu-latest` | 打 `v*` tag 时执行 | 自动创建 GitHub Release 并附带两个二进制 |

流水线内已包含验证：

- aarch64：`file` 校验为 ARM aarch64 可执行文件，并通过 QEMU（`docker --platform linux/arm64`）真实执行 `--version` / `train` / `info` / `predict`，断言输出包含 `__label__`。
- Windows：原生执行 `--version` / `train` / `predict` / `info`，断言输出包含 `__label__`。

本地**不需要**安装交叉编译环境，也不需要在本地编译，直接推送即可：

```bash
git add .
git commit -m "feat: rust + fasttext small model with llm fallback"
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

产物基于较低版本 glibc 交叉编译（cross 镜像），无 OpenSSL/CA 依赖（TLS 使用 rustls + 内置根证书），
所以在纯净的 aarch64 Linux 上可直接运行。

## 本地开发（可选）

```bash
cargo build --release
cargo run --release -- train --input data/train.txt --output model.bin
cargo run --release -- predict --model model.bin --text "vpn tunnel keeps dropping"
```

如需在本地交叉编译 aarch64（需要 Docker）：

```bash
cargo install cross --locked
cross build --release --target aarch64-unknown-linux-gnu
```

## 目录结构

```
src/main.rs         CLI 入口（train / predict / analyze / llm / info）
src/classifier.rs   fastText 小模型封装
src/llm.rs          大模型（OpenAI 兼容接口）客户端与结果解析
data/train.txt      示例训练数据（5 类运维工单）
data/test.txt       示例测试数据
Cross.toml          cross 交叉编译配置
.github/workflows/build.yml  CI 流水线
```

## License

MIT
