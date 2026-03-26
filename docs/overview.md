# Overview

This repo is a systems-oriented set of notes on how LLM workloads actually execute. The throughline is not just model architecture, but the full path from transformer math to hardware behavior: roofline limits, tensor shapes, collective communication, KV-cache growth, serving schedulers, and the compiler/runtime layers that turn graphs into kernels.

Most of the writing is organized around one practical question: what is the real bottleneck? Across the docs, that usually means separating compute-bound from bandwidth-bound work, understanding when communication becomes the scaling limit, and tracing how layout, sharding, batching, and runtime choices change end-to-end performance.

The repo is therefore less a broad survey of "all LLM topics" and more a focused notebook on performance reasoning for inference, serving, distributed execution, and adjacent systems concerns.

## Documents

### Core performance models

- [`docs/roofline.md`](roofline.md) — introduces the repo's basic performance lens: arithmetic intensity, compute vs. memory vs. communication time, and why roofline reasoning is the right starting point for system analysis.
- [`docs/transformer-systems.md`](transformer-systems.md) — breaks transformer cost into projections, core attention, KV-cache growth, FlashAttention, MoE communication, and the rules of thumb that connect model structure to hardware cost.

### Distributed execution and communication

- [`docs/communication.md`](communication.md) — explains the main collectives, why local math can still require global synchronization, and how communication shows up concretely inside tensor-parallel transformer blocks.
- [`docs/tensor-parallelism.md`](tensor-parallelism.md) — gives the compact row-parallel vs. column-parallel mental model for transformer layers and identifies where all-reduce actually appears inside attention and MLP blocks.

### Inference and serving systems

- [`docs/inference-systems.md`](inference-systems.md) — covers prefill vs. decode, tensor shapes during inference, KV-cache reuse, decode-time bottlenecks, latency/throughput heuristics, and common serving-engine layouts.
- [`docs/llm-serving-system-design.md`](llm-serving-system-design.md) — focuses on serving decisions from the GPU execution side: TP/DP/PP trade-offs, KV-cache placement, paged attention, continuous batching, chunked prefill, and prefill/decode disaggregation.

### Implementation and compiler/runtime notes

- [`docs/llama2-cpp.md`](llama2-cpp.md) — uses `llama2.cpp` as an implementation anchor for config-to-shape mapping, RoPE mechanics, KV-cache interpretation, and the practical meaning of weight tying.
- [`docs/deep-learning-compiler.md`](deep-learning-compiler.md) — explains the framework/runtime/compiler/backend split, multi-level IR, layout decisions, graph lowering, and the low-level optimization vocabulary behind deep learning compilers.

### WIP extensions

- [`docs/wip/tpu-systems.md`](wip/tpu-systems.md) — a compact TPU-focused note on MXU/VPU structure, VMEM vs. HBM, pod interconnect hierarchy, and mesh/sharding intuition that still needs a fuller pass.
- [`docs/wip/training-scaling.md`](wip/training-scaling.md) — a training-side summary of data parallelism, FSDP/ZeRO, tensor parallelism, pipeline parallelism, and collective behavior that still needs expansion.
