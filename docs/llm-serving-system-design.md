# LLM Serving System Design Notes

This file collects the previously missed material from the `LLM System Design` sub-page export. It keeps the original focus on practical hardware-aware reasoning for transformer serving and distributed inference.

## GQA and tensor parallelism

### Softmax happens along sequence, not across GPUs

One of the source's repeated clarifications is that, for attention scores with shape `BTSKG`, softmax is applied along the **`S` dimension**.

That means:

- each query head normalizes over key positions in the sequence
- softmax is usually a local operation on one device
- cross-GPU communication is not the same thing as softmax normalization

### What GQA means

Grouped-query attention separates the number of query heads from the number of key/value heads.

- `N` = number of query heads
- `K` = number of KV heads
- `G = N // K` = number of query heads per KV group

The key interpretation preserved from the source:

- query heads are reorganized into `K` groups
- each query group attends only to its corresponding KV head group
- GQA does **not** mean every query head attends to every KV head

### Column-parallel vs. row-parallel inside a transformer block

#### Column parallel

Split weights by output channels.

- each device has the full input activation
- each device produces part of the output channels
- this usually does not need an all-reduce immediately

Typical locations:

- `W_q`
- `W_k`
- `W_v`
- MLP expansion projections such as `W_in`

#### Row parallel

Split weights by input channels and split the input accordingly.

- each device computes a full-shape partial contribution to the final output
- those contributions must be summed
- this is where **all-reduce** appears

Typical locations:

- `W_o`
- `W_out`

### Communication points in one transformer block

The source keeps the block-level flow very clearly:

#### Attention sublayer

Local work:

- Q/K/V projections
- `QK^T`
- masking
- softmax over `S`
- `AV`

Communication:

- row-parallel `W_o`
- all-reduce after `W_o`

#### MLP sublayer

Local work:

- expansion projections
- activation or gating
- elementwise multiplication where applicable

Communication:

- row-parallel output projection
- all-reduce after `W_out`

### KV cache distribution under tensor parallelism

The source notes that KV cache is typically distributed by KV head.

For example, in a TP group:

- each device stores the KV cache for the KV heads it owns
- local query shards attend to local KV cache shards
- communication usually happens only after the row-parallel output projection

## TP, DP, and PP trade-offs for serving

The serving notes preserve one core argument: decode is usually **memory-bound**, so the best deployment choice is not always the highest possible tensor-parallel degree.

### Why decode is memory-bound

The source uses a roofline-style reasoning:

- arithmetic intensity during decode is approximately proportional to batch size
- if batch size stays below the hardware ridge point, the workload remains memory-bound
- in that regime, HBM bandwidth matters more than raw peak FLOPs

### Why TP can become too large

The notes argue that overly large TP can hurt efficiency because:

- per-device matrix slices become too small
- SM occupancy drops due to over-sharding
- all-reduce latency becomes a larger fraction of per-token time

This is the source's explanation for why a configuration such as `TP=4 + DP=2` may outperform pure `TP=8` on throughput, even when pure `TP=8` leaves more memory for KV cache.

### Practical summary preserved from the source

- **TP** lowers per-device weight memory but adds communication inside every layer
- **DP** gives higher concurrency when a single TP group can already fit the model
- **PP** is usually less attractive within a single high-bandwidth node because pipeline bubbles and TTFT penalties become more obvious

## Paged KV cache and memory access costs

The source's paged-KV notes are careful and useful.

### Main costs

Paged KV cache does **not** simply mean that memory access becomes completely uncoalesced.

The more accurate costs are:

- metadata or block-table lookup
- pointer indirection
- reduced streaming locality across blocks
- more irregular gather-like access patterns
- more complex kernel control flow and address generation

### Important nuance

Within one block, accesses can still be well aligned and highly coalesced. The bigger problem is that **across blocks** physical continuity is weaker, which makes prefetching and cache behavior less friendly.

### GPU-level reminders preserved from the source

When thinking about paged KV on GPUs, the important mental model is not only CPU-style cache lines. The notes highlight:

- global memory transaction granularity
- L2 sector or cache-line behavior
- 128-byte alignment
- vectorized load granularity
- whether warp addresses can collapse into a small number of transactions

## Continuous batching

The source frames continuous batching as a decode-stage scheduling strategy.

Main idea:

- after each token-generation step, completed requests leave the active batch
- new waiting requests can be admitted immediately
- the goal is to keep decode throughput high instead of rebuilding batches only at request boundaries

The source also distinguishes control and data placement:

- scheduler state and request metadata are usually host-side
- weights, KV cache, and step tensors are usually GPU-resident

Under tensor parallelism, a request advances through the whole TP group together. It is not independently admitted by only one GPU.

## Chunked prefill

Chunked prefill is one of the most valuable missed sections from the sub-page export.

### Problem it solves

A very long prompt can monopolize compute during prefill and delay many short decode requests, causing visible streaming jitter.

### Core solution

Split a long prefill into smaller chunks, then interleave those chunks with ongoing decode work.

The source's main reasoning is:

- decoder-only models are causal, so earlier chunks do not need to be recomputed when later chunks appear
- later chunks only need to attend to already cached KV from earlier chunks
- online softmax bookkeeping makes chunk-by-chunk accumulation possible without recomputing the full history

### Practical trade-off

Chunked prefill slightly worsens TTFT for the long request, but it can greatly improve tail latency and smoothness for ongoing decode traffic.

## Precision and performance notes preserved from the source

The system-design notes also preserve a useful interview-style comparison:

- **INT4 weight-only** can be excellent for decode because it lowers bandwidth pressure
- the same choice can hurt prefill because dequantization adds compute overhead
- **FP8** is presented as a better balanced option on H100-like hardware when both latency and throughput matter

## Low-level optimization reminders

The source ends with several implementation-oriented takeaways:

- keep `head_dim` in the innermost memory layout position when possible for better coalescing
- decode-specific attention may need split-K style work partitioning because `Q=1` leaves too little work otherwise
- fusing postprocessing such as RoPE into a kernel epilogue can reduce extra HBM traffic

## Review notes

A few sections here use concrete hardware numbers as study heuristics. They should be treated as reasoning scaffolds from the notes rather than as polished benchmark claims.
