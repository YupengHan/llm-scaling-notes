# Experience Bullets

These bullets are derived from the study notes in this repo. They are intentionally factual and conservative.

- Built and refined a local study repository on JAX scaling, transformer systems, TPU/GPU architecture, and distributed communication patterns from private research notes.
- Consolidated multi-source notes on roofline analysis, arithmetic intensity, transformer FLOPs, tensor parallelism, and TPU memory and network behavior into structured technical documentation.
- Documented training-scale trade-offs across data parallelism, FSDP/ZeRO sharding, tensor parallelism, and pipeline parallelism, including collective communication patterns such as all-reduce, reduce-scatter, and all-gather.
- Summarized LLM serving system design topics including KV-cache placement, paged KV cache trade-offs, continuous batching, chunked prefill, and hardware-aware TP/DP/PP decisions.
- Organized implementation-oriented notes on `llama2.cpp`, including RoPE mechanics, KV-cache shape interpretation, and weight tying, into publishable Markdown.
- Preserved technical fidelity while translating mixed-language source notes into natural English and keeping resume-facing material separate from the main study docs.
