# Communication Notes

## Why communication appears repeatedly in the source

Across the root page and the training and system-design sub-pages, communication is one of the main limits on strong scaling. Even when single-device math is efficient, multi-device performance can still fail because communication becomes too visible relative to compute.

## All-gather

The notes use ring all-gather as a baseline mental model.

For 8 GPUs in a ring:

- each rank starts with one chunk
- each round delivers one new chunk
- total rounds: `7`

The source also notes that on NVSwitch systems, the physical implementation may be better than a naive ring because the fabric can accelerate broadcast- or multicast-like collective behavior.

### Where it shows up in the notes

- reconstructing sharded weights in FSDP or ZeRO before a layer executes
- restoring full data from distributed shards
- some tensor-parallel layouts when a full replicated input is temporarily needed

## All-reduce

The source preserves the standard decomposition:

`all-reduce = reduce-scatter + all-gather`

For the 8-GPU ring mental model, this becomes:

- `7` rounds for reduce-scatter
- `7` rounds for all-gather
- `14` rounds total

### What it is doing conceptually

A recurring clarification from the notes is that all-reduce is usually about summing **partial contributions**, not about merging softmax probabilities.

This matters especially in row-parallel layers, where each device computes part of the final output contribution and the group must sum those results.

### Where it shows up in the notes

- data-parallel gradient synchronization
- row-parallel `W_o` in attention blocks
- row-parallel `W_out` in MLP blocks
- sharded matmul cases where local products are only partial sums

## Reduce-scatter

Reduce-scatter combines reduction with sharding.

The training sub-page uses it as the most natural explanation for why FSDP or ZeRO is more memory efficient than full all-reduce:

- devices reduce contributions together
- the final result remains sharded instead of fully replicated

That lowers both memory pressure and communication volume compared with always materializing a full reduced result on every device.

## Communication and scaling

A recurring theme across the source is that communication cost is not uniform:

- in-chip and between-chip communication behave differently
- nearby links and far links have different bandwidth limits
- collective choice interacts with model-parallel layout
- choices that look good at small scale can become unattractive once communication dominates

The source also repeatedly ties communication cost to roofline reasoning: a system may stop scaling well not because the math is wrong, but because memory movement or collectives become the true bottleneck.

## Communication in training vs. serving

### Training-oriented emphasis

The training notes focus on:

- gradient synchronization in DP
- all-gather and reduce-scatter in FSDP or ZeRO
- row-parallel and column-parallel communication patterns in sharded matmuls

### Serving-oriented emphasis

The serving notes focus on:

- all-reduce points inside transformer inference blocks
- KV-cache placement under tensor parallelism
- how communication overhead can offset the memory benefits of large TP groups

## Review note

The source includes some partially developed application examples, especially around MLP communication flow and GPU-serving trade-offs. They are preserved here in summarized form and may still deserve a later, more formal rewrite.
