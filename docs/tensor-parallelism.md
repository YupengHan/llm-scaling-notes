# Tensor Parallelism Notes

This file combines the root-page tensor-parallel notes with the clearer explanations preserved in the `LLM System Design` and `Training how to scale` sub-pages.

## Column parallel linear

Start from:

`Y = XW`

with:

- `X : [BT, Din]`
- `W : [Din, Dout]`
- `Y : [BT, Dout]`

Column parallelism splits `W` by columns:

`W = [W1 | W2 | ...]`

Each device computes its own output shard:

- `Y1 = XW1`
- `Y2 = XW2`

### Key property

Each device needs the full input `X`, but the outputs are already valid output shards, so immediate reduction is not required.

### Typical transformer uses

The source repeatedly places these in column parallel form:

- `W_q`
- `W_k`
- `W_v`
- MLP expansion or gate projections such as `D -> F`

## Row parallel linear

Row parallelism splits the input dimension instead.

`X = [X1 | X2 | ...]`

and:

`W = [W1; W2; ...]`

Each device computes a partial contribution:

- `Z1 = X1W1`
- `Z2 = X2W2`

The final output is the sum of those partial results:

`Y = Z1 + Z2 + ...`

### Key property

Each device only needs part of the input, but the output is incomplete until an all-reduce or equivalent summation step happens.

### Typical transformer uses

The source repeatedly places these in row parallel form:

- attention output projection `W_o`
- MLP down projection `W_out`

## Standard transformer pattern preserved from the source

The notes give the classic tensor-parallel arrangement:

- QKV projection → column parallel
- attention core → local
- output projection `W_o` → row parallel
- MLP expansion → column parallel
- activation or gating → local
- MLP down projection → row parallel

The main reason is simple: keep work local for as long as possible and delay communication to the end of each sublayer or block.

## What all-reduce is actually doing here

One important clarification from the system-design sub-page is that all-reduce is not about merging softmax probabilities across GPUs.

Instead, it usually sums **partial contributions** to the same final hidden-state output.

That is why the most common all-reduce points in a transformer block appear after:

- `W_o`
- `W_out`

## GQA-aware interpretation

The sub-page notes also make the attention layout more explicit:

- query heads are grouped into `K` KV groups with per-group factor `G`
- each Q group attends only to its matching KV group
- local Q, K, and V shards can stay local through `QK^T`, masking, softmax, and `AV`
- communication is usually delayed until the row-parallel output projection

## One-line memory aid from the source

A compact heuristic preserved directly from the notes:

- expand dimension → column parallel
- shrink dimension → row parallel

Examples:

- `D -> NH` and `D -> F` usually fit column parallelism
- `NH -> D` and `F -> D` usually fit row parallelism

## Related review note

The source also mentions Megatron-style model parallelism and sequence parallelism, but only briefly. They are not expanded into separate sections here because the export does not develop them fully.
