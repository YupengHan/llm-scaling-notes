# LLM Serving System Design

These notes focus on how a decoder-only transformer maps onto GPU hardware during inference. The emphasis is on first-principles reasoning that matters in practice: attention layout, tensor-parallel communication, KV-cache management, scheduler behavior, and the latency/throughput trade-offs behind serving decisions.

The goal is to keep the explanations compact, technically grounded, and transferable across serving stacks rather than tied to one framework or one benchmark configuration.

## 1. Notation and attention shapes

A useful notation set for reasoning about transformer serving:

- `B`: batch size
- `L`: number of layers
- `T`: query length
- `S`: key/value sequence length
- `D`: model dimension
- `F`: MLP hidden dimension
- `H`: head dimension
- `N`: number of query heads
- `K`: number of KV heads
- `G = N // K`: number of query heads per KV head in GQA

For an input activation `X` with shape `BTD`:

- `Q = X W_q -> BTNH`
- `K = X W_k -> BSKH`
- `V = X W_v -> BSKH`

Under grouped-query attention, `Q` is naturally viewed as `BTKGH`. The score computation becomes:

```text
BTKGH × BSKH -> BTSKG
```

This makes the grouping rule explicit: for each KV group `k`, only the corresponding query heads in that group participate in attention.

A key serving-time detail is that **softmax is applied along the sequence axis `S`**, not across GPUs. In other words, for each fixed `(b, t, k, g)`, attention normalizes over key positions. That distinction matters because local normalization and cross-device synchronization solve different problems.

## 2. GQA and tensor parallelism

### What GQA changes

Grouped-query attention reduces the number of KV heads without reducing the number of query heads. The practical implication is:

- multiple query heads share one KV head
- each query group attends only to its matching KV group
- the main serving benefit is lower KV-cache memory and lower attention memory pressure

That is why GQA is usually discussed as a **KV-efficiency optimization**, not as a reduction in query-side expressiveness.

### Column-parallel vs. row-parallel linear layers

Tensor parallelism works because large matrix multiplications can be partitioned in ways that preserve correctness.

#### Column parallel

Split the weight matrix by output channels:

```text
Y = XW
W = [W0, W1, ..., Wp-1]
Yi = X Wi
```

Each device:

- receives or already has the full input activation `X`
- computes only its output slice `Yi`
- does not usually need an immediate all-reduce

Typical transformer locations:

- `W_q`
- `W_k`
- `W_v`
- MLP expansion / gating projections such as `W_in1`, `W_in2`

#### Row parallel

Split the weight matrix by input channels and split the input the same way:

```text
Y = XW
W = [W0; W1; ...; Wp-1]
X = [X0, X1, ..., Xp-1]
Yi = Xi Wi
Y = sum_i Yi
```

Each device computes a **full-shape partial contribution** to the final output, so a reduction is required.

Typical transformer locations:

- `W_o`
- `W_out`

This is the cleanest way to think about all-reduce in a transformer block: it is not combining softmax probabilities, and it is not merging attention scores. It is summing partial contributions to the same final hidden vector.

## 3. Where communication happens in one transformer block

A useful mental model is to separate work that is typically local from work that requires synchronization.

### Attention sublayer

Usually local to each tensor-parallel shard:

1. Q/K/V projections
2. `QK^T`
3. mask application
4. softmax over `S`
5. `AV`

Then:

6. row-parallel output projection `W_o`
7. all-reduce across the TP group

### MLP sublayer

Usually local to each shard:

1. expansion projections such as `W_in1` / `W_in2`
2. activation or gating
3. elementwise multiply where applicable

Then:

4. row-parallel output projection `W_out`
5. all-reduce across the TP group

Two details are easy to miss:

- **Local softmax is not the same thing as cheap softmax.** It still scales with sequence length and is often limited by memory bandwidth.
- The natural partitioning differs by sublayer. In attention, it is intuitive to think in terms of heads or head groups. In the MLP, it is more natural to think in terms of the expansion dimension `F`.

## 4. KV cache under tensor parallelism

### Physical placement

Under standard TP for attention, KV cache is usually sharded by KV head. Each device stores the cache for the KV heads it owns, and its local query shard attends to those local KV tensors.

That is why decode often communicates only after the row-parallel output projection rather than during score computation itself.

A practical caveat is that some implementations replicate KV cache in special cases, for example when the effective number of KV heads is smaller than the tensor-parallel degree. The important high-level point is still the same: **KV placement follows the attention partitioning strategy**, and that placement drives both memory usage and communication behavior.

### Why KV cache dominates serving memory

For long-context serving, KV cache often becomes the main memory constraint after weights are loaded. A simple way to think about the growth per generated token is:

```text
KV bytes per token ≈ 2 × L × K × H × dtype_bytes
```

The factor of `2` comes from storing both keys and values. Under GQA, the number of KV heads `K` is smaller than the number of query heads `N`, which is exactly why GQA helps so much at serving time.

## 5. Decode is usually memory-bound

A core serving distinction is:

- **Prefill** is usually more compute-dense and often compute-bound
- **Decode** is usually lower arithmetic intensity and often memory-bound

A simple throughput model is:

```text
throughput ≈ DP × batch_per_group / step_time
```

And a useful decomposition of one decode step is:

```text
step_time ≈ max(compute_time, memory_time) + communication + scheduler_overhead
```

During decode, each linear layer behaves much more like GEMV than GEMM because the query length is tiny, often effectively `T = 1` for the newly generated token. In that regime, the arithmetic intensity is low and weight/KV movement from HBM matters more than peak tensor-core FLOPs.

That leads to an important design conclusion: **maximizing tensor parallelism is not always the best way to maximize throughput**.

### Why larger TP can hurt

Over-sharding can create two problems at once:

1. **smaller per-device matrix slices**
   - lower occupancy
   - worse kernel efficiency
   - more sensitivity to launch overhead and wave quantization effects

2. **more communication steps**
   - all-reduce latency becomes a larger fraction of each decode step
   - synchronization costs become harder to hide

This is why a configuration like `TP=4 + DP=2` can outperform pure `TP=8` on end-to-end throughput even if pure `TP=8` leaves more aggregate memory available for KV cache. The exact crossover depends on the model shape, context length, batch size, and the real efficiency of the kernels and interconnect, but the first-principles logic is stable.

## 6. TP vs. DP vs. PP in serving

| Strategy | What is split | Strength | Main cost | Where it is usually attractive |
|---|---|---|---|---|
| **TP** | tensor dimensions inside layers | lets one model shard fit on multiple GPUs; good intra-node scaling | all-reduce inside the model path | single node with fast GPU interconnect |
| **DP** | full model replicas | highest concurrency once one TP group already fits the model | duplicates weights across replicas | high-throughput serving when memory allows |
| **PP** | layers across stages | useful when the model is too large or interconnect is weaker | bubbles, scheduling complexity, TTFT impact | larger multi-node deployments |

The reason PP is often less attractive inside a single strong NVLink/NVSwitch node is that the bubble cost and first-token latency penalty are more visible, while TP communication is relatively cheap inside the box.

## 7. Paged KV cache: what it fixes and what it costs

Paged KV cache is worth understanding precisely because it solves a real systems problem: memory fragmentation.

### Why it is useful

A serving engine wants to admit requests with very different prompt lengths and generation lengths. Preallocating one giant contiguous KV region per request wastes memory badly. Paging fixes that by turning KV allocation into block allocation.

### The real overheads

The common oversimplification is: “paged KV makes memory access uncoalesced.” That is incomplete.

The more accurate costs are:

- block-table or metadata lookup
- pointer indirection and address generation
- weaker streaming locality across blocks
- more gather-like access patterns when a sequence spans many blocks
- more complex kernel control flow

The nuance that matters is this: **access inside one block can still be aligned and well coalesced**. The bigger problem is that physical continuity across blocks is weaker, so large-range streaming, prefetching, and cache behavior become less friendly.

### GPU-level implications

When reasoning about paged KV on GPU, the most relevant concepts are:

- memory-transaction granularity
- L2 sector or cache-line behavior
- 128-byte alignment
- vectorized load width
- whether warp addresses collapse into a small number of memory transactions

Implementation-wise, a good paged-attention kernel tries to preserve locality within a block, stage page metadata efficiently, and tile the computation so that irregularity does not dominate the whole kernel.

## 8. Continuous batching

Continuous batching is a **decode-stage scheduling policy** rather than a one-time batching trick.

At each generation iteration:

- finished requests leave the active batch
- waiting requests can be admitted immediately
- the engine tries to keep the decode batch saturated without waiting for the whole batch to finish

Two practical details matter:

1. Scheduler state and request metadata usually live on the host side.
2. Weights, KV cache, and per-step tensors live on the GPU.

Under tensor parallelism, a request advances through the entire TP group together. A new request is not admitted by just one GPU; it is admitted into the next decode step for the whole TP group.

## 9. Chunked prefill

Chunked prefill exists to protect interactive decode traffic from long prompts.

### The problem

A very long prompt can launch a large prefill workload that monopolizes compute and delays the next decode step for many active users. The user-visible symptom is streaming jitter.

### The idea

Split a long prompt into smaller chunks and interleave those chunks with decode work.

This works because decoder-only transformers are causal:

- earlier tokens do not depend on later tokens
- once a prefill chunk has produced its hidden states and written its KV tensors, that work is complete
- later chunks only need to attend to already cached historical KV plus their own local chunk

### Why this does not require recomputing everything

The key technical trick is the same online-softmax logic used in tiled attention. Instead of materializing one giant score matrix, the kernel keeps running softmax statistics as it processes tiles or chunks.

```text
m_new = max(m_past, m_local)
l_new = l_past * exp(m_past - m_new) + l_local * exp(m_local - m_new)
```

That lets the engine combine contributions from previous KV blocks and the current chunk without redoing the entire attention history.

### The trade-off

Chunked prefill typically:

- slightly increases TTFT for the long request
- significantly reduces jitter and tail latency for ongoing decode traffic
- improves overall scheduler fairness in mixed workloads

This is a good example of a serving optimization that is not about raw FLOPs. It is about protecting the latency profile of the whole system.

## 10. Prefill / decode disaggregation

At larger scale, the difference between prefill and decode can justify architectural separation.

- **Prefill** prefers hardware and kernels that are optimized for large, compute-dense matrix multiplications.
- **Decode** prefers hardware, layouts, and schedulers that maximize memory efficiency and steady-state token throughput.

That is why some serving systems split these phases into separate pools:

1. a prefill path builds the initial KV cache
2. the resulting KV state is handed off to a decode path
3. the decode path focuses on continuous batching and low-jitter generation

This is not always worth doing inside a small single-node deployment, because the transfer and coordination overhead can outweigh the gain. But at cluster scale, prefill/decode disaggregation is a natural extension of the same reasoning behind chunked prefill: the two phases stress the hardware differently and benefit from different optimization goals.

## 11. Precision choices in serving

Precision selection is not just a memory question. It changes where the bottleneck lands.

### BF16

A strong baseline because it is simple, stable, and well supported.

### INT4 weight-only

Very attractive for decode because it cuts weight bandwidth pressure sharply. But weight-only quantization can hurt prefill if dequantization overhead becomes significant during compute-heavy GEMMs.

### FP8

On hardware with native FP8 support, FP8 is often a more balanced serving choice than aggressive weight-only quantization because it reduces memory pressure while keeping the compute path friendlier for both prefill and decode. The right answer still depends on model quality tolerance, kernel maturity, and calibration strategy.

## 12. Low-level optimization patterns that actually matter

A few bottom-up optimizations show up repeatedly in fast serving stacks:

### Memory layout for coalescing

For attention KV tensors, it is often beneficial to keep `head_dim` in the innermost position so that warp reads are contiguous and vectorized loads are possible.

### Decode-specific attention kernels

During decode, `Q` is tiny, often just one token. That means standard attention parallelization can underutilize the GPU. A common fix is a split-K or flash-decoding style strategy that parallelizes over long KV history instead of over the query dimension.

### Kernel fusion

Fusing post-processing such as RoPE application, bias handling, scaling, or simple epilogue transforms avoids extra round-trips to HBM. In serving, reducing memory traffic is often more valuable than shaving a few scalar operations.

## 13. Common pitfalls I try to avoid

A few distinctions are easy to blur when discussing serving systems:

- **Softmax direction vs. communication pattern**: softmax is along sequence length; all-reduce is about summing partial outputs.
- **Output shard vs. partial contribution**: column parallel produces an output shard; row parallel produces a full-shape partial sum.
- **Local operation vs. cheap operation**: being local to one GPU does not mean the kernel is inexpensive.
- **More TP vs. better performance**: higher tensor-parallel degree helps memory fit, but not necessarily latency or throughput.
- **Paged memory vs. broken coalescing**: paging makes locality harder, but the real story is more nuanced than “everything becomes random.”

## 14. Practical takeaways

The main lessons I would carry into a serving design discussion are:

- GQA matters primarily because it reduces KV-cache cost.
- The most common TP communication points are after `W_o` and `W_out`.
- Decode performance is often governed by HBM bandwidth, cache layout, and communication overhead rather than by peak FLOPs.
- Continuous batching and chunked prefill are scheduler-level tools for protecting system-wide latency, not just average throughput.
- Paged KV cache improves memory efficiency, but only works well with kernels that respect GPU memory-access rules.
- The best serving setup is usually the one that balances model fit, communication cost, scheduler behavior, and hardware utilization at the same time.
