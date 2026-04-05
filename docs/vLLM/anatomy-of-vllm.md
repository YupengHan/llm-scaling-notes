# High-Throughput LLM Inference Systems

## Overview

This note covers the core internals of a production-grade LLM inference engine (based on vLLM architecture), organized into the following topics:

1. **LLM Engine** — scheduling, paged attention, continuous batching
2. **Advanced Features** — chunked prefill, prefix caching, guided & speculative decoding, disaggregated prefill/decode
3. **Scalability** — single GPU → multi-GPU
4. **Benchmarks & Auto-tuning** — throughput and latency metrics

---

## ToDo

- `EngineCoreRequest: cache_salt`
- `SOM — Structured Output Manager`
- `DeepSeek-V2 MLA`
- `KV Cache C++ Design`

---
