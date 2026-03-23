# JAX Scale Learning

A local-only, refined repository version of a private Notion study tree about JAX scaling, transformer systems, TPU/GPU performance, distributed training, and LLM serving design.

## Scope

This repo now treats the original root export and the three sub-page exports as one coherent note set:

- `jax-scaling-book.zip`
- `jax-scaling-book_LLM-System-Design.zip`
- `jax-scaling-book_Training-how-to-scael.zip`
- `jax-scaling-book_llma2CPP.zip`

The goal of this refinement pass is to preserve the original structure and wording as much as possible while filling in missing material from the sub-page exports, normalizing the notes into natural English, fixing grammar, and keeping unsupported additions to a minimum.

## What is in this repo

### Core study notes

- `docs/overview.md` — scope, priorities, and the core framing from the root page
- `docs/roofline.md` — roofline thinking, arithmetic intensity, BF16, and communication rooflines
- `docs/transformer-systems.md` — transformer FLOPs, attention structure, GQA, MoE, KV cache, and FlashAttention notes
- `docs/tpu-systems.md` — TPU architecture, memory hierarchy, pod networking, and mesh/sharding notes
- `docs/communication.md` — collective communication patterns and where they appear in model systems
- `docs/tensor-parallelism.md` — column-parallel vs. row-parallel patterns and transformer block communication points

### New material integrated from missed sub-page exports

- `docs/training-scaling.md` — data, tensor, FSDP/ZeRO, and pipeline parallel training notes
- `docs/llm-serving-system-design.md` — TP/DP/PP trade-offs, KV cache, paged KV, continuous batching, chunked prefill, and decoding-oriented systems notes
- `docs/llama2-cpp.md` — config interpretation, RoPE notes, and weight tying notes from the `llama2.cpp` export

### Supporting files

- `docs/source-manifest.md` — archive-by-archive source manifest and mapping notes
- `resume/experience-bullets.md` — resume-facing derivative bullets kept separate from the study docs
- `assets/images/` — selected images kept only where they materially support the notes

## Editing approach

- kept everything local only
- did not create a GitHub remote
- did not push anything
- refined the existing repo in place rather than replacing it from scratch
- preserved the original wording and study sequence where possible
- added the previously missed sub-page content conservatively
- normalized mixed Chinese/English source notes into natural English
- fixed grammar, spelling, and Markdown structure where needed
- flagged unclear or underspecified items rather than inventing content

## Important improvements in this pass

Compared with the earlier pass, this repo now explicitly incorporates the three exported sub-pages that were previously only referenced from the root notes:

- **LLM System Design** now has its own document with serving-oriented system design notes
- **Training how to scale** now has its own document with training parallelism and sharded-array communication notes
- **Llama2.cpp** now has its own document covering configuration, RoPE, and weight tying
- the source manifest now tracks all four archives instead of only the root archive
- the README no longer treats the root page as the only meaningful source

## Notes for review

Some source sections are still intentionally conservative because the exported notes themselves remain outline-like or partially ambiguous. In particular, these areas may need your review if you want a deeper second pass:

- `Part_7 Inference` from the root page is still mostly a pointer rather than a full standalone chapter
- `Part_12 GPUs` remains a short outline rather than a fully developed note set
- the numeric placeholders `6`, `8`, and `11` are still not clearly defined in the source
- some formulas and rule-of-thumb serving estimates in the LLM systems notes should be treated as study notes, not production benchmarks
