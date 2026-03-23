# Inference Systems Notes

This document integrates the newly exported `Inference` sub-page into the existing repo. It keeps the original focus on hardware-aware reasoning for decoder-only LLM inference, while cleaning up grammar and translating the mixed Chinese/English notes into natural English.

## Core framing: prefill vs. generation

The source repeatedly separates inference into two stages:

- **Prefill**: run the full prompt through the model once and build the KV cache.
- **Generation / decode**: generate one new token at a time using the accumulated KV cache.

For a decoder-only transformer with causal self-attention:

- during **prefill**, query length `T` and key/value length `S` are both the prompt length
- during **generation**, the current query length is `T = 1`, while `S` is the length of the cached history

This split works because future tokens do not modify the representations of past tokens. Once a past token's K/V has been computed in a given layer, it can be reused safely.

## Tensor shapes for a 70B-style example

The source uses a Llama-style 70B example with these architectural constants:

- `D = 8192` — model dimension
- `N = 64` — number of query heads
- `K = 8` — number of key/value heads
- `H = 128` — head dimension, where `D / N = 128`
- `G = 8` — number of query heads per KV head, where `N / K = 8`
- `F = 28672` — MLP hidden dimension
- `V = 128256` — vocabulary size
- `L = 80` — number of layers

### Fixed weight shapes

These weight shapes stay fixed regardless of whether the model is in prefill or generation:

- `W_Q: [D, N*H] = [8192, 8192]`
- `W_K: [D, K*H] = [8192, 1024]`
- `W_V: [D, K*H] = [8192, 1024]`
- `W_O: [N*H, D] = [8192, 8192]`
- `W_In1: [D, F] = [8192, 28672]`
- `W_In2: [D, F] = [8192, 28672]`
- `W_Out: [F, D] = [28672, 8192]`

### Prefill-stage activations

For a prompt length of 2048:

- input `X [B, T, D] = [B, 2048, 8192]`
- projected `Q [B, T, N, H] = [B, 2048, 64, 128]`
- projected `K, V [B, S, K, H] = [B, 2048, 8, 128]`
- GQA reshape `Q [B, T, K, G, H] = [B, 2048, 8, 8, 128]`
- attention scores `QK^T [B, T, S, K, G] = [B, 2048, 2048, 8, 8]`
- attention output before merge `[B, 2048, 8, 8, 128]`
- merged attention output `[B, 2048, 64, 128]`
- output projection `[B, 2048, 8192]`

For the MLP:

- input `[B, 2048, 8192]`
- gate and up projections: both `[B, 2048, 28672]`
- activated and multiplied intermediate `[B, 2048, 28672]`
- down projection `[B, 2048, 8192]`

A practical note preserved from the source: large score tensors are often discussed conceptually, but high-performance kernels such as FlashAttention do not fully materialize those huge matrices in HBM.

### Generation-stage activations

If the cache already contains 2048 historical tokens and the model is generating token 2049:

- current input `X [B, 1, 8192]`
- projected `Q [B, 1, 64, 128]`
- cached and concatenated `K, V [B, 2048, 8, 128]`
- reshaped `Q [B, 1, 8, 8, 128]`
- attention scores `[B, 1, 2048, 8, 8]`
- attention output before merge `[B, 1, 8, 8, 128]`
- merged output `[B, 1, 64, 128]`
- final output projection `[B, 1, 8192]`

For the MLP during generation:

- input `[B, 1, 8192]`
- gate and up projections: both `[B, 1, 28672]`
- activated intermediate `[B, 1, 28672]`
- down projection `[B, 1, 8192]`

The source emphasizes the main systems implication: generation computes only one new token, but it must repeatedly read a large KV cache. That is one reason decode is often memory-bandwidth-bound.

## Why KV cache helps

### Naive decoding

Without KV cache, generating token `i` recomputes the full forward pass over all previous tokens plus the new one.

Preserving the source's reasoning:

- FFN work accumulates like `1 + 2 + ... + n = O(n^2)`
- attention work accumulates like `1^2 + 2^2 + ... + n^2 = O(n^3)`

### KV-cached decoding

With KV cache:

- prefill computes the prompt once and stores K/V for every layer
- each decode step computes only the new token's projections and MLP work
- historical K/V is read from cache instead of recomputed

Then:

- FFN work becomes `O(n)`
- attention work becomes `1 + 2 + ... + n = O(n^2)`

### Common clarifications preserved from the source

#### Why cache K/V but not Q?

`Q` is used only for the current token and then discarded. `K` and `V` are the persistent memory that future tokens must repeatedly attend to.

#### Is cache stored only for the first layer?

No. Every transformer layer has its own attention operation, so every layer needs its own cached historical K/V.

#### Why not cache hidden states instead?

Because future attention directly consumes historical `K` and `V`. If only hidden states were stored, each new decode step would still need to recompute historical `K` and `V` using `W_K` and `W_V`.

#### Why pass logits between stages instead of storing only the token?

The source's argument is that logits are the decision basis for sampling. They are needed for strategies such as top-k, top-p, temperature scaling, and beam search. In practice, logits are short-lived intermediate outputs, not long-lived cache like K/V.

## Transformer bottlenecks during inference

The source frames transformer inference around two operator families.

### 1. Linear operations

This includes:

- Q/K/V/O projections
- MLP matrix multiplications
- other dense projection layers

The simplified model in the notes is:

- matrix multiplication `BD * DF = BF`
- FLOPs approximately `2BDF`
- BF16 memory traffic approximately `2BD + 2DF + 2BF`

The source uses this to build a rule of thumb:

- during **prefill**, long sequence length `T` often provides enough work to become compute-bound even without large batch size
- during **generation**, dense projections may need batching to achieve high hardware utilization

A careful nuance preserved from the source: a threshold such as `B >= 240` is only a rough decode-stage rule of thumb for some dense matmuls on some hardware. It is not a universal transformer law.

### 2. Attention operations

The source gives an arithmetic-intensity model for attention:

- read `Q` with shape `[B, T, D]`
- read cached `K/V` with shape `[B, S, D]`
- compute `QK^T` and `AV`
- write output `[B, T, D]`

This leads to the source's simplified multi-head attention arithmetic intensity:

`MAAI = ST / (S + T)`

Main implications:

- **prefill**: when `S = T`, arithmetic intensity grows roughly like `T / 2`, so attention can become compute-bound for sufficiently long sequences
- **generation**: when `T = 1`, arithmetic intensity stays near `1`, so decode attention is typically memory-bandwidth-bound

## Generation latency model

The source includes a useful rule-of-thumb latency model for decode.

In small-batch generation, the minimum per-step time can be approximated as:

`(batch_size * KV_cache_size + parameter_size) / total_memory_bandwidth`

The notes then refine this with a more general interpretation:

- the attention side remains bandwidth-bound
- the MLP side may be compute-bound or bandwidth-bound, depending on batch size and hardware
- small batch gives lower latency but poorer utilization
- larger batch raises efficiency, but it can also increase per-step latency

These are best treated as study heuristics rather than polished production formulas.

## Throughput and latency optimization ideas

The source lists several ways to reduce inference cost.

### GQA

Grouped-query attention reduces the number of KV heads so multiple query heads share the same KV heads. Compared with full multi-head attention, this reduces KV-cache size in proportion to the query-to-KV head ratio.

### Local attention

If some layers use local attention instead of full global attention, the effective KV-cache length for those layers is capped by a smaller context window. In long-context settings, this can materially reduce total KV-cache size.

### Sharing KV across layers

Cross-layer KV sharing can reduce total KV-cache storage. The source also notes an important caveat: it does not automatically improve step time, because the shared cache may still need to be read multiple times from HBM.

### Quantization

The notes argue that inference is often less sensitive than training to reduced precision for parameters and KV cache. Quantizing weights and KV cache to formats such as INT8, INT4, or FP8 can:

- reduce memory-bandwidth demand
- lower the batch size needed to reach the compute roofline
- free memory for larger batches or more active requests

### Ragged HBM and paged attention

Paged attention avoids large amounts of global padding in KV-cache allocation. It is a runtime systems optimization rather than a model-architecture change. The benefit is better memory utilization; the cost is more implementation complexity and less regular access behavior.

## Distributing inference

The source distinguishes prefill and generation because their best scaling strategies differ.

### Prefill scaling strategy

The notes argue for this order of operations:

1. start with **tensor parallelism / model parallelism** to distribute weights and large matmuls
2. continue increasing TP until interconnect communication becomes the bottleneck
3. if more devices are still available and sequence length remains large, extend further with **sequence parallelism**

Main motivations for TP during prefill:

- the model may not fit on one device
- long-sequence prefill contains large, high-intensity GEMMs that benefit from more aggregate FLOPs

### Generation scaling strategy

Generation is described as harder to optimize because:

- it is harder to gather large batches of live requests
- latency targets are stricter
- the stage is more memory-bound and more communication-sensitive

The source's practical principles are:

- move activations when needed, but avoid moving full KV cache or full parameters unnecessarily
- with larger batches, increase model parallelism until the system hits a FLOPs-versus-interconnect limit
- with smaller batches, stronger model sharding can reduce latency, even if throughput falls slightly
- if model-parallel sharding exceeds the number of KV heads, KV cache may also need to be partitioned along the batch dimension

## Inference-engine design patterns

One of the strongest parts of the new export is its comparison of serving layouts.

### 1. Monolithic batching: prefill together, then generate together

In this design, a whole batch is prefetched together and then generated together.

Advantages:

- conceptually simple

Disadvantages preserved from the source:

- time-to-first-token can become poor when prefill batches are large
- short requests can be delayed by long requests during generation
- prefill suffers from padding to the longest prompt in the batch
- prefill and generation are forced to share the same sharding strategy, even though their optimal layouts may differ

### 2. Interleaved prefill and decode

The source then describes a more dynamic pattern:

- prefill can run with `B = 1` or in smaller chunks
- decode requests remain batched
- prefill and decode are interleaved at scheduling time

Benefits:

- better TTFT because prefill no longer waits for a large shared batch
- decode can still preserve reasonable throughput

Trade-offs:

- while a large prefill is executing, decode traffic may stall unless prefill is chunked
- inter-token latency can become less smooth
- TTFT improves, but streaming smoothness may worsen

### 3. Disaggregated prefill and decode

The source presents this as a modern large-scale serving pattern.

Main idea:

- prefill runs on a prefill pool
- decode runs on a decode pool
- KV cache is handed off from prefill to decode

Benefits preserved from the source:

- lower latency at scale because requests do not contend as directly across the two phases
- independent scaling of prefill capacity and decode capacity
- better hardware specialization, since prefill and decode can use different sharding strategies and even different machine pools

This matches the source's recurring systems point: prefill is more compute-oriented, while decode is more memory- and latency-oriented.

## Continuous batching, chunked prefill, and mixed workloads

The export also asks an implementation-oriented question: should prefill and decode run together?

The source's conclusion is nuanced:

- for the **same request**, prefill and decode are inherently sequential
- across **different requests**, many real systems do interleave prefill and decode on the same GPUs

The notes specifically connect this to:

- continuous batching / in-flight batching
- chunked prefill
- dynamic scheduling rather than fixed static resource partitioning

The key idea is not simply to run "one CUDA stream for prefill" and "one CUDA stream for decode." The real problem is scheduling around shared contention for:

- SMs
- registers and shared memory
- L2 cache
- HBM bandwidth
- model weights
- growing KV cache

So the practical optimization target is scheduler design, not a simplistic two-stream split.

## Open items left conservative

A few source items are still intentionally left light:

- the export references specific diagrams and screenshots that were not required to preserve the core technical meaning in text
- the `Prefix Caching` and `JetStream` headings appear in the source but are not developed there
- a linked LLaMA 13B throughput/latency example appears as an external reference rather than self-contained notes

Those areas may be worth a future pass if you want the inference section expanded further.
