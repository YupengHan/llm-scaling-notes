# Transformer Systems Notes

## FLOPs basics

The source starts from standard contraction examples:

- vector dot product: `2P` FLOPs
- matrix-vector multiply: `2NP` FLOPs
- matrix-matrix multiply: `2NPM` FLOPs

![Contracting and batching](../assets/images/matmul-contracting-and-batching.png)

It then uses these to explain why training is roughly three times the forward-pass cost for the same matmul:

- forward pass: `2NPM`
- gradient with respect to weights: `2NPM`
- gradient with respect to activations: `2NPM`

Total during training:

`6NPM`

This leads to the well-known rule of thumb preserved in the notes:

`training FLOPs per token ≈ 6 × parameter count`

## Transformer cost structure

![Transformer dimension graph](../assets/images/transformer-dimension-graph.png)

The source breaks the transformer into a few core contractions:

1. **QKV projection**: project from model dimension `D` into attention-head structure
2. **attention scores**: contract over head dimension `H`
3. **attention over values**: contract over sequence dimension `S`

The notes explain these contractions in plain language rather than only as equations, and that intent is preserved here.

## Why attention is split into projection vs. core attention

The source distinguishes two parts of the attention block:

### 1. Projection matmuls

These include `Q`, `K`, `V`, and output projection `O`.

Their properties in the source notes:

- they use learnable parameters
- they are token-local linear transforms
- they do not yet perform token-to-token interaction

### 2. Core attention

This includes:

- `QK^T`
- softmax
- attention-weighted `V`

Their properties in the source notes:

- no new learned parameter matrices are introduced here
- token-to-token interaction happens here
- cost scales strongly with sequence length
- this is why long context is expensive and why kernels such as FlashAttention matter

## GQA and dot-product attention

A useful clarification preserved from the source is that these labels describe different things:

- **GQA / MHA / MQA** describe how query heads relate to KV heads
- **dot-product attention** describes how attention scores are computed

So a layer can simultaneously be:

- causal self-attention
- grouped-query attention
- scaled dot-product attention

The source also explains the grouping identity:

`N = K * G`

where query heads can be viewed as KV groups times per-group query heads.

## Short-context training rule of thumb

For dense transformers at reasonable context lengths, the source records this approximation:

`FLOPs = (18BTDF + 12BTD(N + K)H)L`

and rewrites it into the familiar summary:

`≈ 6 × total tokens × total parameters`

The notes explicitly warn that this is a rule of thumb, not a universal law.

## Attention share vs. matmul share

The source gives the ratio:

`attention FLOPs / matmul FLOPs = T / (8D)`

under simplifying assumptions such as `F = 4D`, `D = NH`, and `N = K`.

The point is not the exact formula alone, but when attention becomes large enough relative to the rest of the layer to matter operationally.

## Sparse models and MoE

The notes keep a concise MoE summary:

- each layer has `E` expert MLP blocks
- each token activates only `k` of them
- total parameter count grows roughly with `E`
- activated parameters per token grow with `k`
- MoE introduces two all-to-all operations: one entering experts and one leaving them

This preserves the source emphasis that MoE changes both compute structure and communication structure.

## Key takeaways preserved from the source

### KV cache

The source highlights that KV cache size scales approximately with:

`2 * S * L * K * H`

and stresses that reducing KV heads through GQA or MQA directly reduces KV cache size.

### FlashAttention

The source emphasizes that FlashAttention does **not** remove the quadratic attention FLOPs. Instead, its value comes from:

- avoiding materializing the full attention matrix
- processing `K` and `V` in tiles
- keeping running statistics and outputs in on-chip memory where possible
- reducing memory pressure and raising arithmetic intensity

### Rematerialization / gradient checkpointing

The source frames rematerialization as a direct memory-vs.-compute trade-off:

- save memory
- pay extra recomputation FLOPs

### Why matmul matters so much

One key line preserved from the notes is that matmul compute scales cubically while data movement scales quadratically, so larger matmuls are more likely to reach compute saturation.
