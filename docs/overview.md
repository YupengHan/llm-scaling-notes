# Overview

This repository is a curated collection of my learning notes on LLM systems, originally developed in Notion and later reorganized into a repo-first format.

Rather than trying to be a polished textbook or a broad survey of everything in modern AI, this repo is meant to show how I think through LLM systems from first principles: how model architecture interacts with hardware, where training and inference costs come from, how communication and memory become bottlenecks, and how system design choices should be made under real constraints.

Across these notes, I focus on questions such as:

- how GPUs and TPUs execute LLM workloads
- how transformer computations map to hardware limits
- how to estimate compute, memory, and communication cost
- how to reason about parallelism strategies for training and inference
- how to identify the actual bottleneck before optimizing the wrong metric

A recurring theme throughout the repo is that benchmark gains matter less if they come with equivalent losses in system efficiency. Many of the notes use roofline reasoning and bottleneck analysis to connect model behavior with hardware realities such as memory bandwidth, interconnect cost, and device utilization.

This repo is organized around a few core systems topics:

- roofline modeling
- transformer and attention cost analysis
- TPU / GPU system fundamentals
- training and scaling tradeoffs
- inference and serving design
- implementation-oriented notes such as `llama2.cpp`

If you are skimming the repo, the best entry points are:

- `docs/roofline.md`
- `docs/transformer-systems.md`
- `docs/inference-systems.md`
- `docs/communication.md`
- `docs/llm-serving-system-design.md`
- `docs/deep-learning-compiler.md`

Overall, this repository is intended to demonstrate:

- structured self-driven learning across the LLM stack
- systems-oriented reasoning instead of isolated paper summaries
- the ability to move between theory, hardware constraints, and implementation tradeoffs
- a habit of turning raw research notes into organized technical documentation

## Current layout

Most reader-facing notes live directly under `docs/`. Active work that still needs a fuller pass now lives under `docs/wip/`.

Reader-facing docs currently include:

- `docs/roofline.md`
- `docs/transformer-systems.md`
- `docs/inference-systems.md`
- `docs/communication.md`
- `docs/tensor-parallelism.md`
- `docs/llm-serving-system-design.md`
- `docs/llama2-cpp.md`
- `docs/deep-learning-compiler.md`

Current WIP docs:

- `docs/wip/tpu-systems.md`
- `docs/wip/training-scaling.md`

## Progress snapshot

The git history now shows two clear phases of work:

- March 22, 2026: the initial public doc set landed across roofline, transformer systems, tensor parallelism, serving, TPU notes, training notes, and implementation notes
- March 23-25, 2026: deeper passes landed for inference systems, communication, `llama2.cpp`, repo framing, and the new deep learning compiler note

The current explicit TODO bucket is:

- `docs/wip/tpu-systems.md`
- `docs/wip/training-scaling.md`

Other older backlog ideas have not been started yet. They remain follow-up work rather than active in-progress edits.

The repo is still evolving, but the core focus is stable: understanding LLM training and inference as end-to-end systems problems, not just modeling problems.
