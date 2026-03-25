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
- [`docs/communication.md`](docs/communication.md) — collective communication baselines, application-level communication flow, and scaling constraints.
- [`docs/llm-serving-system-design.md`](docs/llm-serving-system-design.md) — serving layouts, batching strategies, paged KV cache, and decode-oriented system trade-offs.
- [`docs/deep-learning-compiler.md`](docs/deep-learning-compiler.md) — framework/runtime/compiler/backend roles, IR levels, graph lowering, and hardware-aware optimization passes.
- [`docs/llama2-cpp.md`](docs/llama2-cpp.md) — implementation-oriented reasoning about config interpretation, RoPE, KV cache, and weight tying.

## Work history and progress

- `March 22, 2026`: the initial public doc set landed, including roofline, transformer systems, serving notes, tensor-parallel notes, TPU notes, training notes, and implementation notes.
- `March 23-25, 2026`: deeper passes landed for [`docs/inference-systems.md`](docs/inference-systems.md), [`docs/communication.md`](docs/communication.md), and [`docs/llama2-cpp.md`](docs/llama2-cpp.md), and the new [`docs/deep-learning-compiler.md`](docs/deep-learning-compiler.md) was added.
- `Current WIP`: [`docs/wip/tpu-systems.md`](docs/wip/tpu-systems.md) and [`docs/wip/training-scaling.md`](docs/wip/training-scaling.md).
- `Backlog not started`: deeper follow-up work for [`docs/tensor-parallelism.md`](docs/tensor-parallelism.md), fuller serving expansions in [`docs/llm-serving-system-design.md`](docs/llm-serving-system-design.md), and a small amount of optional visual cleanup.

## Repo map

### Core systems and performance notes

- [`docs/overview.md`](docs/overview.md) — deeper framing for the repo, its systems lens, and how the material is organized.
- [`docs/roofline.md`](docs/roofline.md) — roofline reasoning, arithmetic intensity, and communication-aware performance limits.
- [`docs/transformer-systems.md`](docs/transformer-systems.md) — transformer FLOPs, attention structure, KV cache, and model-side cost drivers.
- [`docs/deep-learning-compiler.md`](docs/deep-learning-compiler.md) — compiler/runtime/backend separation, IR levels, data layout, and lowering/optimization flow.
- [`docs/communication.md`](docs/communication.md) — collective communication patterns and where they appear in distributed model systems.
- [`docs/tensor-parallelism.md`](docs/tensor-parallelism.md) — tensor-parallel execution patterns and communication points inside transformer blocks.

### Inference and serving focus

- [`docs/inference-systems.md`](docs/inference-systems.md) — inference-stage behavior from KV-cache reuse to decode latency and throughput constraints.
- [`docs/llm-serving-system-design.md`](docs/llm-serving-system-design.md) — system design trade-offs for batching, cache management, and serving architecture.
- [`docs/llama2-cpp.md`](docs/llama2-cpp.md) — implementation-oriented notes on config interpretation, RoPE, KV cache, and weight tying.

### Work in progress

- [`docs/wip/tpu-systems.md`](docs/wip/tpu-systems.md) — TPU architecture, memory hierarchy, networking, and mesh-based execution notes awaiting a fuller pass.
- [`docs/wip/training-scaling.md`](docs/wip/training-scaling.md) — training parallelism and collective-communication notes awaiting a deeper pass.

### Supporting material

- [`resume/experience-bullets.md`](resume/experience-bullets.md) — resume-facing bullets distilled from the technical work in this repo.
- [`assets/images/`](assets/images/) — diagrams referenced throughout the notes.

## Notes on scope

- This is a technical study and documentation repository, not a production inference framework or benchmark suite.
- Depth varies by topic, but the throughline is systems reasoning under hardware and execution constraints.
- Quantitative heuristics in the notes are for performance reasoning and trade-off analysis, not production guarantees.
