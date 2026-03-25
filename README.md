# LLM Scaling Notes

A self-study repository on LLM inference, model systems, scaling, and the compiler/runtime ideas that connect model graphs to hardware. The material is centered on JAX/ML scaling topics and extended with serving, communication, hardware-focused notes, and a first-pass deep learning compiler writeup that is directly relevant to `LLM Inference Performance Engineer` work.

## Project Goal

This repo is meant to serve as a public technical study portfolio. It documents how I think about:

- inference bottlenecks across prefill and decode
- roofline analysis, arithmetic intensity, and bandwidth limits
- transformer design choices that affect serving cost and latency
- KV-cache layout, memory pressure, and batching trade-offs
- tensor, data, and pipeline parallelism plus collective communication
- TPU/GPU architecture, memory hierarchy, and interconnect behavior
- framework/runtime/compiler/backend separation, IR levels, graph lowering, and hardware-aware optimization choices

## Scope

The repository started from private study notes based on JAX scaling material and has been reorganized into a cleaner public set of Markdown docs. It is not a production inference framework or a benchmark suite. The emphasis is on systems understanding, design trade-offs, performance-oriented reasoning, and the compiler/runtime concepts that connect graph-level model descriptions to real hardware execution.

## Current Status

This is an active study repository, not a finished handbook. Some docs are already solid first-pass writeups, while others are still partial, conservative, or only lightly expanded from the original notes.

The strongest current coverage is:

- roofline reasoning and arithmetic intensity
- transformer cost structure, KV cache, and attention-related bottlenecks
- tensor parallelism, communication collectives, and serving trade-offs
- inference-stage concepts such as prefill vs. decode, batching, and KV-cache-driven performance limits
- deep learning compiler stack concepts such as framework/runtime/compiler/backend roles, IR levels, data layout, operator fusion, tiling, and backend-specific optimization

## Repo Map

### Core systems and performance notes

- `docs/overview.md` — scope, priorities, the study roadmap, and currently known gaps
- `docs/roofline.md` — roofline thinking, arithmetic intensity, BF16, and communication rooflines
- `docs/transformer-systems.md` — transformer FLOPs, attention structure, GQA, MoE, KV cache, and FlashAttention notes
- `docs/deep-learning-compiler.md` — framework/runtime/compiler/backend roles, high-level vs. low-level IR, data layout, and frontend/backend optimization passes
- `docs/tpu-systems.md` — compact first-pass notes on TPU architecture, memory hierarchy, pod networking, and mesh/sharding
- `docs/communication.md` — collective communication patterns and where they appear in model systems
- `docs/tensor-parallelism.md` — column-parallel vs. row-parallel patterns and transformer block communication points

### Inference and serving focus

- `docs/inference-systems.md` — prefill vs. decode, KV cache, inference bottlenecks, batching, and serving-engine design patterns
- `docs/llm-serving-system-design.md` — TP/DP/PP trade-offs, paged KV cache, continuous batching, chunked prefill, and decoding-oriented system design notes
- `docs/llama2-cpp.md` — implementation-oriented notes on config interpretation, RoPE, KV cache, and weight tying

### Training and supporting material

- `docs/training-scaling.md` — data, tensor, FSDP/ZeRO, and pipeline parallel training notes
- `resume/experience-bullets.md` — resume-facing bullets derived from the study work
- `assets/images/` — diagrams used where they materially support the notes

## ToDo

- expand the unfinished root-source areas tracked in [`docs/overview.md`](docs/overview.md), especially `Part_12 GPUs`, the link-heavy transformer architecture section, and the still-light sections `Part_6 Training LLaMA`, `Part_8 Serving LLaMA`, and `Part_11 Conclusions`
- deepen [`docs/inference-systems.md`](docs/inference-systems.md) by turning `Prefix Caching`, `JetStream`, and the external latency/throughput references into self-contained notes
- expand the brief mentions of Megatron-style model parallelism and sequence parallelism in [`docs/tensor-parallelism.md`](docs/tensor-parallelism.md) and [`docs/training-scaling.md`](docs/training-scaling.md)
- rewrite the partially developed application examples in [`docs/communication.md`](docs/communication.md), especially MLP communication flow and GPU-serving trade-offs
- optionally add a few missing visuals, such as a MoE diagram, where they improve understanding rather than just decorate the notes
- deepen [`docs/deep-learning-compiler.md`](docs/deep-learning-compiler.md) with more concrete examples from XLA, TVM, TensorRT, and TorchInductor, and tighten the distinction between graph-level optimization and loop/kernel-level optimization

## What To Expect

- These are study notes and technical writeups, not polished production docs.
- Not every topic in the repo is equally complete yet; the open gaps are tracked above in `ToDo`.
- Some sections remain conservative where the original material was outline-like or ambiguous.
- Quantitative rules of thumb in serving sections should be read as study material, not as validated production benchmarks.
