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
- `Question: [Chunk Prefill Idea]`
  - `could the scheduler estimate work by KV-cache load / memory traffic, rather than only by the current number of generated tokens?`
- `Question: [Prefix Caching]`
  - `exact prefix matching is a very strict requirement. For many web-search or document-heavy cases, there may be large middle spans that are identical even when the prompt is not identical from the beginning. For example, if a 16k-token webpage contains many repeated 1k-token blocks, there can still be a lot of redundant computation even though standard prefix caching cannot reuse it directly. How could that case be optimized?`
  - `More broadly, taking raw webpage text, tokenizing it, and then converting it into KV cache still feels like an inefficient path. Is there a better abstraction?`
  - `Another thought: instead of treating a conversation as one single thread from beginning to end, could we treat it as multiple fresh conversations so repeated webpage KV becomes reusable? The remaining problem is how to make use of several conversations that each restart from the beginning.`
  - `Even without prefix caching, storing KV cache is itself a memory-for-compute trade-off.`
  - `Since KV cache stores precomputed key/value states rather than final attention outputs, is there any analogous idea to online softmax that would let part of the attention work be reused while the rest is updated incrementally?`
- "Accelerating Large Language Model Decoding with Speculative Sampling": https://arxiv.org/abs/2302.01318
- n-gram
- EAGLE "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" https://arxiv.org/abs/2401.15077
- Medusa Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads https://arxiv.org/abs/2401.10774

---
