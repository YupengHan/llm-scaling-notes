# Overview

This repo is a systems-oriented set of notes on how LLM workloads actually execute. The throughline is not just model architecture, but the full path from transformer math to hardware behavior: roofline limits, tensor shapes, collective communication, KV-cache growth, serving schedulers, concrete inference-engine internals, and the compiler/runtime layers that turn graphs into kernels.

Most of the writing is organized around one practical question: what is the real bottleneck? Across the docs, that usually means separating compute-bound from bandwidth-bound work, understanding when communication becomes the scaling limit, and tracing how layout, sharding, batching, and runtime choices change end-to-end performance.

The repo is therefore less a broad survey of "all LLM topics" and more a focused notebook on performance reasoning for inference, serving, distributed execution, and adjacent systems concerns.

## Documents

The notes are grouped by folder, so the easiest way to read the repo is to treat each directory as a small topic cluster.

### `docs/jax-ml-scaling/`

This folder is the main numbered path for scaling notes: roofline first, then TPU hardware context, then transformer execution, and later inference behavior.

- [`docs/jax-ml-scaling/1_roofline.md`](jax-ml-scaling/1_roofline.md) — roofline basics for reasoning about compute, memory bandwidth, and communication bottlenecks.
- [`docs/jax-ml-scaling/2_tpu.md`](jax-ml-scaling/2_tpu.md) — TPU architecture, memory hierarchy, slice/Pod networking, and the performance intuition behind VMEM, ICI, and DCN.
- [`docs/jax-ml-scaling/4_transformer.md`](jax-ml-scaling/4_transformer.md) — transformer cost structure, attention/MLP behavior, KV-cache growth, and system-level rules of thumb.
- [`docs/jax-ml-scaling/7_inference.md`](jax-ml-scaling/7_inference.md) — prefill vs. decode, KV-cache reuse, and the main latency/throughput trade-offs in inference engines.
- [`docs/jax-ml-scaling/5_training.md`](jax-ml-scaling/5_training.md) — training-side scaling note covering DP, TP, PP, and collective behavior; still in progress.

### `docs/vLLM/`

This folder zooms in on one concrete inference stack: how a production-style engine is decomposed into request processing, scheduling, KV-cache management, model execution, and RPC-facing control layers, plus the concrete per-step mechanics of batching and forward execution.

- [`docs/vLLM/anatomy-of-vllm.md`](vLLM/anatomy-of-vllm.md) — vLLM-oriented engine anatomy note covering `EngineCoreRequest`, scheduler behavior, paged KV-cache blocks, model execution, and the role of RPC in larger deployments.
- [`docs/vLLM/llm-inference-engine.md`](vLLM/llm-inference-engine.md) — a more detailed inference-engine walkthrough covering request lifecycle, paged KV-cache allocation, continuous batching, chunked prefill, scheduling constraints, distributed inference, and the forward-pass pipeline.

### `docs/ml-systems-practice/`

This folder is more implementation-facing: concrete serving design choices, communication/sharding notes, and code-oriented model notes.

- [`docs/ml-systems-practice/communication.md`](ml-systems-practice/communication.md) — collective communication basics and where synchronization shows up in tensor-parallel transformer blocks.
- [`docs/ml-systems-practice/tensor-parallelism.md`](ml-systems-practice/tensor-parallelism.md) — row-parallel vs. column-parallel intuition and where all-reduce appears inside attention and MLP layers.

- [`docs/ml-systems-practice/llm-serving-system-design.md`](ml-systems-practice/llm-serving-system-design.md) — practical serving-system design notes around TP/DP/PP, KV-cache layout, paging, batching, and scheduler trade-offs.
- [`docs/ml-systems-practice/llama2-cpp.md`](ml-systems-practice/llama2-cpp.md) — implementation notes from `llama2.cpp`, focused on config mapping, RoPE, KV-cache layout, and weight tying.

### `docs/compiler/`

This folder holds compiler/runtime notes for the path from model graph to hardware execution.

- [`docs/compiler/deep-learning-compiler.md`](compiler/deep-learning-compiler.md) — framework/runtime/compiler/backend roles, IR lowering, layout choices, and common low-level optimization ideas.

### `docs/gpu/`

This folder is for hardware-specific GPU notes.

- [`docs/gpu/hopper.md`](gpu/hopper.md) — Hopper features that matter for kernel work, especially clusters, DSMEM, TMA, and async barriers.

### `docs/triton/`

This folder is for custom-kernel notes built around Triton and Hopper-era execution details.

- [`docs/triton/triton_vecadd_tmaload.md`](triton/triton_vecadd_tmaload.md) — a hands-on Triton note using vector add and block pointers to explain launch grids, memory access, and TMA-style loading.
- [`docs/triton/wip_triton_overview_en.md`](triton/wip_triton_overview_en.md) — WIP overview of Triton, blocked programming, clusters, descriptors, and Hopper-oriented kernel reasoning.
