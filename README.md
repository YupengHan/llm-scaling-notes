# LLM Scaling Notes

A curated collection of LLM systems notes, reorganized from private study work into a public repository. The repo focuses on inference, training and scaling, serving, communication, GPU/TPU systems, and compiler/runtime concepts, with an emphasis on hardware-aware reasoning, bottleneck analysis, and performance trade-offs.

## What this repo covers

- roofline analysis, arithmetic intensity, and bandwidth limits
- transformer cost structure, attention behavior, and KV-cache bottlenecks
- prefill vs. decode performance behavior and inference-stage constraints
- tensor, data, pipeline, and serving-time parallelism plus collective communication
- TPU/GPU architecture, memory hierarchy, and interconnect-aware reasoning
- serving design trade-offs such as batching, cache management, and request scheduling
- compiler/runtime/backend concepts that connect model graphs to hardware execution

## Recommended starting points

- [`docs/roofline.md`](docs/roofline.md) — start here for roofline thinking, arithmetic intensity, and bandwidth-limited performance.
- [`docs/transformer-systems.md`](docs/transformer-systems.md) — transformer compute structure, attention cost, GQA, KV cache, and FlashAttention-related reasoning.
- [`docs/inference-systems.md`](docs/inference-systems.md) — prefill vs. decode, latency/throughput heuristics, and inference bottlenecks.
- [`docs/training-scaling.md`](docs/training-scaling.md) — training-side scaling strategies across data, tensor, FSDP/ZeRO, and pipeline parallelism.
- [`docs/tpu-systems.md`](docs/tpu-systems.md) — TPU architecture, memory hierarchy, pod topology, and mesh/sharding concepts.
- [`docs/llm-serving-system-design.md`](docs/llm-serving-system-design.md) — serving layouts, batching strategies, paged KV cache, and decode-oriented system trade-offs.
- [`docs/deep-learning-compiler.md`](docs/deep-learning-compiler.md) — framework/runtime/compiler/backend roles, IR levels, graph lowering, and hardware-aware optimization passes.

## Repo map

### Core systems and performance notes

- [`docs/overview.md`](docs/overview.md) — deeper framing for the repo, its systems lens, and how the material is organized.
- [`docs/roofline.md`](docs/roofline.md) — roofline reasoning, arithmetic intensity, and communication-aware performance limits.
- [`docs/transformer-systems.md`](docs/transformer-systems.md) — transformer FLOPs, attention structure, KV cache, and model-side cost drivers.
- [`docs/deep-learning-compiler.md`](docs/deep-learning-compiler.md) — compiler/runtime/backend separation, IR levels, data layout, and lowering/optimization flow.
- [`docs/tpu-systems.md`](docs/tpu-systems.md) — TPU architecture, memory hierarchy, networking, and mesh-based execution.
- [`docs/communication.md`](docs/communication.md) — collective communication patterns and where they appear in distributed model systems.
- [`docs/tensor-parallelism.md`](docs/tensor-parallelism.md) — tensor-parallel execution patterns and communication points inside transformer blocks.

### Inference and serving focus

- [`docs/inference-systems.md`](docs/inference-systems.md) — inference-stage behavior from KV-cache reuse to decode latency and throughput constraints.
- [`docs/llm-serving-system-design.md`](docs/llm-serving-system-design.md) — system design trade-offs for batching, cache management, and serving architecture.
- [`docs/llama2-cpp.md`](docs/llama2-cpp.md) — implementation-oriented notes on config interpretation, RoPE, KV cache, and weight tying.

### Training and supporting material

- [`docs/training-scaling.md`](docs/training-scaling.md) — training parallelism strategies and scaling trade-offs across the stack.
- [`resume/experience-bullets.md`](resume/experience-bullets.md) — resume-facing bullets distilled from the technical work in this repo.
- [`assets/images/`](assets/images/) — diagrams referenced throughout the notes.

## Notes on scope

- This is a technical study and documentation repository, not a production inference framework or benchmark suite.
- Depth varies by topic, but the throughline is systems reasoning under hardware and execution constraints.
- Quantitative heuristics in the notes are for performance reasoning and trade-off analysis, not production guarantees.
