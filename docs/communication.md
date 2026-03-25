# Communication Notes

## Why communication keeps showing up

Across training, serving, and system-design notes, communication is one of the main limits on strong scaling. Even when single-device math is efficient, multi-device performance can still flatten because communication becomes visible relative to compute.

Two recurring patterns matter:

- the math on each GPU may be correct locally, but still incomplete globally
- the collective that stitches local results together can become the real bottleneck

## Collective cheat sheet

| Collective | 8-GPU ring baseline | What it does | Common LLM usage |
| --- | --- | --- | --- |
| `all-gather` | `7` rounds | rebuilds a full tensor from sharded chunks | reconstructing sharded weights, restoring full activations, some TP layouts |
| `reduce-scatter` | `7` rounds | reduces contributions while keeping the result sharded | ZeRO/FSDP-style sharded reductions, sharded outputs consumed by the next stage |
| `all-reduce` | `14` rounds (`reduce-scatter + all-gather`) | sums partial contributions and leaves every rank with the same result | DP gradient sync, row-parallel attention output, row-parallel MLP output |

## All-gather

The notes use **ring all-gather** as the baseline mental model.

For 8 GPUs in a ring:

```text
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
```

- each rank starts with one chunk
- each round delivers one new chunk
- total rounds: `7`

This is the simplest way to reason about a collective: every GPU gradually receives the missing chunks until the full tensor is reconstructed.

### Where it shows up in the notes

- reconstructing sharded weights in FSDP or ZeRO before a layer executes
- restoring full data from distributed shards
- some tensor-parallel layouts when a full replicated input is temporarily needed

## Reduce-scatter

Reduce-scatter is the natural **sum-and-keep-sharded** primitive.

Instead of first summing contributions and then materializing the full reduced tensor on every GPU, reduce-scatter combines the reduction with the final partitioning step.

For the same 8-GPU ring mental model:

- total rounds: `7`

### Why it matters

- it lowers memory pressure relative to fully replicated reduction
- it lowers communication relative to always rebuilding a full tensor everywhere
- it is the first half of the standard high-performance formulation of all-reduce

### Where it shows up in the notes

- ZeRO or FSDP explanations of sharded gradient/state updates
- sharded reductions where the next stage can consume partitioned output directly
- the first stage of a high-performance all-reduce

## All-reduce

Conceptually, all-reduce is often explained as **reduce + broadcast**.

In practice, the performance formulation is:

```text
all-reduce = reduce-scatter + all-gather
```

For the 8-GPU ring baseline, this becomes:

- `7` rounds for reduce-scatter
- `7` rounds for all-gather
- `14` rounds total

### What it is doing conceptually

A useful clarification from the notes is that all-reduce is usually about summing **partial contributions**, not about merging softmax probabilities or combining unrelated outputs.

This matters especially in row-parallel layers, where each device computes one valid partial contribution to the same final tensor and the group must sum those contributions.

### Where it shows up in the notes

- data-parallel gradient synchronization
- row-parallel `W_o` in attention blocks
- row-parallel `W_2` / `W_out` in MLP blocks
- sharded matmul cases where local products are only partial sums

## Application-level communication: tensor-parallel MLP

A clean example is the transformer MLP:

```text
X:   [N, H]      input activations
W1:  [H, 4H]     expansion projection
W2:  [4H, H]     projection back to hidden size

Z = GeLU(X · W1) · W2
```

Where:

- `N` = number of tokens
- `H` = hidden size

### Step 1: column-parallel `W1` (no communication on the forward local path)

Split `W1` **along columns** across 8 GPUs.

- GPU `i` holds `W1_i` with shape `[H, 4H/8]`
- each GPU computes its local shard:

```text
Y_i = X · W1_i
[N, H] · [H, 4H/8] = [N, 4H/8]
```

Because `GeLU` is element-wise, each GPU can apply the activation locally:

```text
A_i = GeLU(Y_i)
```

### Why there is no communication here

After the column split, each GPU owns a disjoint slice of the expanded hidden dimension. Since `GeLU` is element-wise, nothing needs to be exchanged before the activation is applied.

That is the important intuition: **column-parallel first projection + element-wise activation does not force a collective on the forward path.**

### Step 2: row-parallel `W2` (local matmul first, then collective)

Now split `W2` **along rows** across 8 GPUs.

- GPU `i` holds `W2_i` with shape `[4H/8, H]`
- GPU `i` uses its local activation shard `A_i` and computes:

```text
Z_i = A_i · W2_i
[N, 4H/8] · [4H/8, H] = [N, H]
```

Notice the subtle point: each `Z_i` already has the full output width `[N, H]`, but it is **not** the final answer. It is only GPU `i`'s **partial sum contribution** to the final output.

By block matrix multiplication:

```text
Z = Z_1 + Z_2 + ... + Z_8
```

So the group must perform an all-reduce to obtain the final `Z`.

### The communication takeaway from the MLP example

- the `W1` column-parallel stage is communication-free on the forward local compute path
- the activation stays local because it is element-wise
- the `W2` row-parallel stage creates **partial sums**
- those partial sums require an **all-reduce**

A precise way to say it is:

> the local row-parallel matmul itself does not communicate, but the layer output is incomplete until the all-reduce sums the per-GPU contributions

This is one of the clearest examples of **application communication** inside a transformer block.

## Ring baseline vs. NVSwitch-aware implementations

The notes use ring collectives as the baseline mental model because they are easy to reason about.

### Ring baseline

For 8 GPUs:

- ring all-gather: `7` rounds
- ring reduce-scatter: `7` rounds
- ring all-reduce: `7 + 7 = 14` rounds

This baseline is useful for intuition, capacity planning, and understanding why collectives become visible as tensor-parallel group size grows.

### NVSwitch-aware note

On NVSwitch systems, the actual runtime may be better than a literal naive ring.

The notes specifically call out two ideas:

- transfers are routed through the **NVSwitch fabric**, not a dedicated point-to-point GPU cable
- the hardware/runtime can accelerate **broadcast- or multicast-like** collective behavior

That means a ring is still a good mental model, but it is not always the exact physical execution strategy.

### Optimized all-reduce paths

The notes also mention NVSwitch-aware optimized all-reduce paths such as **TensorRT-LLM MultiShot**.

The intuition is:

- instead of always paying the full `2N - 2` ring schedule
- the system can use a **reduce-scatter-like ownership phase**
- followed by a **multicast all-gather-like phase**

Under that mental model, the whole collective can complete in **2 communication steps** on the right NVSwitch-aware stack.

The practical takeaway is not that ring theory is wrong. It is that **ring is the baseline mental model, while specialized fabrics and runtimes can do better in practice.**

## Communication and scaling

A recurring theme across the notes is that communication cost is not uniform:

- in-chip and between-chip communication behave differently
- nearby links and far links have different bandwidth/latency behavior
- collective choice interacts with model-parallel layout
- choices that look good at small scale can become unattractive once communication dominates

The notes also repeatedly tie communication to roofline reasoning: a system may stop scaling well not because the math is wrong, but because memory movement or collectives become the true bottleneck.

## Communication in training vs. serving

### Training-oriented emphasis

The training notes focus on:

- gradient synchronization in data parallelism
- all-gather and reduce-scatter in FSDP or ZeRO
- row-parallel and column-parallel communication patterns in sharded matmuls

### Serving-oriented emphasis

The serving notes focus on:

- all-reduce points inside transformer inference blocks
- KV-cache placement under tensor parallelism
- how communication overhead can offset the memory benefits of large TP groups
- how hardware-aware collectives can materially change real serving latency
