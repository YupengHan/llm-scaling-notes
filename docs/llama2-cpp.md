# Llama2.cpp Notes

This file preserves and lightly cleans the implementation-oriented notes from the `Llama2.cpp` sub-page export.

## Config interpretation

The source starts by mapping config fields to transformer notation.

- `D = config.dim`
- `F = config.hidden_dim`
- `L = config.n_layers`
- `N = config.n_heads`
- `K = config.n_kv_heads`
- `V = config.vocab_size`
- `H = D / N`
- `B = 1` in the referenced code path
- `T = 1` for each forward step in decode
- `S = current cache length = pos + 1`, with an upper bound of `config.seq_len`

The notes also distinguish between physical storage shape and logical interpretation:

- `q`, `k`, and `v` may be stored physically as `[D]`, but are interpreted logically as `[N, H]`
- `key_cache` and `value_cache` may be stored physically as `[L, seq_len, D]`, but are easier to reason about as `[L, S, N, H]`

## RoPE: rotary position embedding

### Basic form

RoPE rotates every two dimensions of a head vector as a 2D plane.

For one 2D block:

\[
R(lpha) =
egin{bmatrix}
\cos lpha & -\sin lpha \
\sin lpha & \cos lpha
\end{bmatrix}
\]

The source highlights the useful identity:

\[
R(lpha)^T = R(-lpha)
\]

### Applying RoPE to Q and K

If query `q` comes from position `m` and key `k` comes from position `n`, then after RoPE:

\[
q' = R(m	heta)q
\]
\[
k' = R(n	heta)k
\]

The resulting attention score becomes:

\[
(q')^T k' = q^T R((n-m)	heta) k
\]

### Main conclusion

This is the key intuition preserved from the notes:

- RoPE naturally encodes **relative position**
- the attention score depends on `n - m`, not just on an absolute position ID

### Multi-dimensional view

For a full head vector, RoPE is not one giant rotation.
Instead:

- every two dimensions form one 2D block
- each block is rotated independently
- together they form a block-diagonal rotation structure

The notes also preserve the usual frequency intuition:

- earlier dimension pairs use higher-frequency rotations
- later pairs use lower-frequency rotations

## Weight tying

The source explains weight tying as sharing the same parameter matrix between:

- the input token embedding table
- the final output projection used to produce logits

### Why it works conceptually

The same matrix plays two roles:

1. map token IDs into the model's hidden space
2. compare the final hidden state against the vocabulary space again at output time

The notes emphasize that the output-side matmul can be viewed as measuring similarity between the final hidden state and each token embedding.

### Benefits preserved from the source

- saves a large amount of parameter memory
- reduces extra computation for a separate output head
- acts as a useful regularizer by tying input and output representations together

## Conservative review notes

These notes are intentionally close to the source and remain focused on conceptual understanding rather than line-by-line code commentary.
