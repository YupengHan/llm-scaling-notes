# LLM Scaling Notes

A curated collection of LLM systems notes, reorganized from private study work into a public repository. The repo focuses on inference, training and scaling, serving, concrete engine internals, communication, GPU/TPU systems, and compiler/runtime concepts, with an emphasis on hardware-aware reasoning, bottleneck analysis, and performance trade-offs. Recent updates expand the vLLM serving notes with speculative decoding variants such as n-gram lookup, EAGLE, and Medusa. For the full document guide, start with [`docs/overview.md`](docs/overview.md).

## Checkpoint

### Done

| File | Last updated |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | `2026-04-11` |
| [`docs/jax-ml-scaling/1_roofline.md`](docs/jax-ml-scaling/1_roofline.md) | `2026-03-22` |
| [`docs/jax-ml-scaling/2_tpu.md`](docs/jax-ml-scaling/2_tpu.md) | `2026-04-01` |
| [`docs/jax-ml-scaling/4_transformer.md`](docs/jax-ml-scaling/4_transformer.md) | `2026-03-22` |
| [`docs/jax-ml-scaling/7_inference.md`](docs/jax-ml-scaling/7_inference.md) | `2026-03-25` |
| [`docs/vLLM/anatomy-of-vllm.md`](docs/vLLM/anatomy-of-vllm.md) | `2026-04-10` |
| [`docs/vLLM/advance-features.md`](docs/vLLM/advance-features.md) | `2026-04-11` |
| [`docs/vLLM/llm-inference-engine.md`](docs/vLLM/llm-inference-engine.md) | `2026-04-04` |
| [`docs/ml-systems-practice/communication.md`](docs/ml-systems-practice/communication.md) | `2026-03-25` |
| [`docs/ml-systems-practice/llm-serving-system-design.md`](docs/ml-systems-practice/llm-serving-system-design.md) | `2026-03-25` |
| [`docs/ml-systems-practice/llama2-cpp.md`](docs/ml-systems-practice/llama2-cpp.md) | `2026-03-25` |
| [`docs/compiler/deep-learning-compiler.md`](docs/compiler/deep-learning-compiler.md) | `2026-03-25` |
| [`docs/triton/triton_vecadd_tmaload.md`](docs/triton/triton_vecadd_tmaload.md) | `2026-03-29` |

### In Progress

| File | Last updated |
| --- | --- |
| [`docs/jax-ml-scaling/5_training.md`](docs/jax-ml-scaling/5_training.md) | `2026-03-25` |
| [`docs/triton/wip_triton_overview_en.md`](docs/triton/wip_triton_overview_en.md) | `2026-03-29` |
| [`docs/gpu/hopper.md`](docs/gpu/hopper.md) | `2026-03-27` |

### To Do

| File | Last updated |
| --- | --- |
| [`docs/ml-systems-practice/tensor-parallelism.md`](docs/ml-systems-practice/tensor-parallelism.md) | `2026-03-22` |
