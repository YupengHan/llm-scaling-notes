# Llama2.cpp Notes

This file preserves and lightly cleans the implementation-oriented notes from the `Llama2.cpp` sub-page export. It keeps the original structure while updating several details from the source notes, especially the config-to-shape mapping, the RoPE derivation, and the practical meaning of weight tying in `run.cpp`.

## Config interpretation

The source starts by mapping config fields to transformer notation.

- `D = config.dim`
- `F = config.hidden_dim`
- `L = config.n_layers`
- `N = config.n_heads`
- `K = config.n_kv_heads`
  - Conceptually, this is the KV-head count.
  - In the referenced minimal code path, however, the tensors and cache are not laid out the way a full modern GQA implementation usually would be, so `K` is best treated here as a conceptual field rather than something that is fully realized in every shape below.
- `V = config.vocab_size`
- `H = D / N`
- `B = 1` in the referenced code path (single-batch decode)
- `T = 1` for each forward step in autoregressive decode (one current token per call)
- `S = current cache length = pos + 1`, with an upper bound of `config.seq_len`
- `config.seq_len` is the maximum context length or KV-cache capacity, not the current `S` at every step

The original notes also preserve the code skeleton directly:

```cpp
struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    // weights for rmsnorms
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    // weights for attention matmuls
    tensor3d wq;  // [layer, dim, dim]
    tensor3d wk;  // [layer, dim, dim]
    tensor3d wv;  // [layer, dim, dim]
    tensor3d wo;  // [layer, dim, dim]
    // weights for ffn
    tensor3d w1;  // [layer, hidden_dim, dim]
    tensor3d w2;  // [layer, dim, hidden_dim]
    tensor3d w3;  // [layer, hidden_dim, dim]
    // final rmsnorm
    tensor1d rms_final_weight;  // [dim]
    // freq_cis for RoPE relatively positional embeddings
    tensor2d freq_cis_real;  // [seq_len, (dim/n_heads)/2]
    tensor2d freq_cis_imag;  // [seq_len, (dim/n_heads)/2]
};
```

A small but important nuance: the `vocab_size` comment above comes from the referenced snippet and should be read as a property of that code path, not as a claim about all Llama-family tokenizers.

The notes also distinguish between physical storage shape and logical interpretation:

- `q`, `k`, and `v` may be stored physically as `[D]`, but are interpreted logically as head-structured tensors such as `[N, H]`
- `key_cache` and `value_cache` may be stored physically as `[L, seq_len, D]`, but are easier to reason about as `[L, S, N, H]`
- this physical-versus-logical separation is useful because the implementation flattens tensors for simple C-style memory layout, while the math is naturally organized per head

## RoPE: rotary position embedding

### Basic form

RoPE rotates every two dimensions of a head vector as a 2D plane.

For one 2D block:

\[
R(\alpha) =
\begin{bmatrix}
\cos \alpha & -\sin \alpha \\
\sin \alpha & \cos \alpha
\end{bmatrix}
\]

The source highlights the useful identity:

\[
R(\alpha)^T = R(-\alpha)
\]

This is another way of seeing that each block is an orthogonal rotation: it preserves the geometry of that 2D subspace instead of arbitrarily distorting it.

### Applying RoPE to Q and K

If query `q` comes from position `m` and key `k` comes from position `n`, then after RoPE:

\[
q' = R(m\theta)q
\]
\[
k' = R(n\theta)k
\]

The attention score can then be expanded as:

\[
(q')^T k' = q^T R(m\theta)^T R(n\theta) k
\]

Using `R(m\theta)^T = R(-m\theta)`, this becomes:

\[
(q')^T k' = q^T R(-m\theta)R(n\theta)k
\]

and because rotations compose additively,

\[
(q')^T k' = q^T R((n-m)\theta) k
\]

### Main conclusion

This is the key intuition preserved from the notes:

- RoPE naturally encodes **relative position**
- the attention score depends on `n - m`, not just on an absolute position ID
- because the transformation is rotational, it preserves the local geometry of each 2D block while injecting position information

### Multi-dimensional view

For a full head vector, RoPE is not one giant rotation. Instead:

- every two dimensions form one 2D block
- each block is rotated independently
- together they form a block-diagonal rotation structure

If the rotary dimension is `d`, then there are `d / 2` such rotation blocks. In a toy full-vector example with `d = 4096`, that would be `2048` blocks. In most practical implementations, though, `d` refers to the dimension that actually participates in rotation, typically `head_dim` or `rotary_dim`, not necessarily the full model width `D`.

The notes also preserve the usual frequency intuition. A common expression is:

\[
\theta_i = 10000^{-\frac{2(i-1)}{d}}, \qquad i \in [1, 2, \dots, d/2]
\]

where `d` is the rotated dimension for that implementation.

The intuition is:

- earlier dimension pairs use higher-frequency rotations
- later dimension pairs use lower-frequency rotations
- different parts of the representation therefore have different sensitivity to changes in position

## Weight tying

The source explains weight tying as sharing the same parameter matrix between:

- the input token embedding table
- the final output projection used to produce logits

In the referenced `run.cpp`-style flow, the same matrix is used twice:

```cpp
copy(state.x, transformer_weights.token_embedding_table[token_index]);
matmul(state.logits, state.x, transformer_weights.token_embedding_table);
```

Conceptually, the first use maps a discrete token ID into hidden space, while the second compares the final hidden state against every token embedding row to produce vocabulary logits.

### Why it works conceptually

The same matrix plays two roles:

1. map token IDs into the model's hidden space
2. compare the final hidden state against the vocabulary space again at output time

The notes emphasize that the output-side matmul can be viewed as measuring similarity between the final hidden state and each token embedding. If the final hidden state points in a direction similar to a particular token vector, the dot product with that row is larger, so that token receives a higher logit.

This is why the weight-tied view is often described as a symmetry of the same semantic space:

- on the input side, the matrix turns token IDs into continuous representations
- on the output side, the model asks which token embedding best matches the final state

### Benefits preserved from the source

- saves a large amount of parameter memory by removing the need for a second large vocabulary-sized matrix
- can improve training behavior by acting as a useful regularizer: the model is encouraged to keep input and output token representations aligned
- can reduce parameter movement and storage overhead, although the logits matmul itself still has to be computed

## Conservative review notes

These notes are intentionally close to the source and remain focused on conceptual understanding rather than line-by-line code commentary. The main updates here are precision improvements: clarifying conceptual versus physical tensor shapes, spelling out the RoPE score derivation, and making the weight-tying explanation more explicit about how the shared embedding table is reused in inference.
