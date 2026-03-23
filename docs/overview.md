# Overview

## Source shape

The study tree is not a single polished manuscript. It is a multi-export Notion note set made up of:

- one root page export
- three sub-page exports that were previously only referenced from the root page
- a mix of outlines, equations, rules of thumb, screenshots, implementation notes, and interview-style reasoning

The four source archives are now treated as one coherent set:

- `jax-scaling-book.zip`
- `jax-scaling-book_LLM-System-Design.zip`
- `jax-scaling-book_Training-how-to-scael.zip`
- `jax-scaling-book_llma2CPP.zip`

## Priority roadmap

### Highest priority

- `0_Intro`
- `1_Roofline`
- `4_Transformer`
- `7_Inference`
- `12_GPU`
- `9_ProfilingTPU`

### Secondary priority

- `3_SharedMatMul`
- `10_ProgrammingTPUinJAX`
- `5_Training`

### Later-pass priority

- `2_TPU`
- `8`
- `6`
- `11`

## Core goals stated in the source

The original introduction centers on a few recurring questions:

- how TPUs and GPUs work internally
- how devices communicate through collectives such as all-gather
- how LLMs run on hardware
- what it costs to train large models
- how much memory inference and deployment require
- how to estimate the gap between current performance and hardware limits
- how to choose parallelism strategies at different scales
- how to design algorithms that match hardware characteristics
- how to identify real bottlenecks before overvaluing benchmark gains

## Main organizing idea

A central source claim is that benchmark improvements are not very meaningful if they come with equivalent losses in roofline efficiency.

The note set repeatedly returns to the same reasoning loop:

1. understand hardware limits first
2. identify whether a workload is compute-bound, memory-bound, or communication-bound
3. reason about transformer cost at both operator and system scales
4. choose parallelism, memory layout, and scheduling strategies that fit those limits

## How this repo is organized

The repo still follows the original study flow, but it now includes the previously missed sub-page material:

- overview and roadmap material stays here
- roofline material lives in `docs/roofline.md`
- transformer and attention cost notes live in `docs/transformer-systems.md`
- TPU hardware notes live in `docs/tpu-systems.md`
- communication and tensor-parallel notes are split into dedicated docs
- training-scale notes from the sub-page export now live in `docs/training-scaling.md`
- LLM serving and system-design notes from the sub-page export now live in `docs/llm-serving-system-design.md`
- implementation-oriented `llama2.cpp` notes now live in `docs/llama2-cpp.md`

## Sections still needing review

These parts remain intentionally conservative because the source is still outline-like or ambiguous in those areas:

- `Part_7 Inference` from the root export is still not a full standalone chapter
- `Part_12 GPUs` is still brief in the source
- `Transformer architecture` appears mostly as external links
- the unlabeled items `8`, `6`, and `11` are still unclear
- some hardware-number examples in the serving notes should be treated as study heuristics rather than polished benchmarks
