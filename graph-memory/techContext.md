# Tech Context

技术栈、依赖与关键实现细节。

### 技术栈：Rust + C++ + Python 原型

type: `component` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/techContext.md`

Core: C++17, CMake 3.16+, Rust 2021, Python 3。
Libtorch/PyTorch 2.11.0, tch-rs 0.24.0（可选 tch-backend）。
QUIC: quinn 0.11 + rustls 0.23 + rcgen 0.13。
模型权重：safetensors, tokenizers, half。

_updated: 2026-06-29 05:34:19_
