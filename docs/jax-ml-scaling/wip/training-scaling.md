# Training Scaling Notes

This file preserves and lightly cleans the training-focused material from the `Training how to scale` sub-page export.

## Main scaling modes

### Data parallelism

In data parallelism, each device holds a full copy of the model, and different replicas process different batches.

#### Strengths

- forward propagation usually does not need communication
- backward communication is important, but it is not the only bottleneck in the critical path
- batch size can be increased naturally, especially during training

#### Limits

- model weights still sit in each device's HBM, so data parallelism does not reduce model-memory pressure
- with BF16 parameters and FP32 Adam optimizer state, the source notes use the rule of thumb that the maximum model size per device is roughly `HBM per device / 10`

## FSDP / ZeRO sharding

The source frames FSDP or ZeRO as a response to large-model training memory limits.

Core idea:

- parameters are stored in sharded form
- before a layer's matmul, devices all-gather the full weight they need
- after use, the full copy does not need to remain resident everywhere

This reduces per-device memory pressure compared with plain data parallelism.

## Tensor parallelism

Tensor parallelism places one layer's large weights across multiple devices so they jointly execute one logical layer.

The source emphasizes three common cases:

- linear layers split by row or column
- attention split across head dimensions
- MLP split across hidden dimensions

## Pipeline parallelism

Pipeline parallelism splits the model by layers into multiple stages, and multiple micro-batches flow through those stages like a pipeline.

The notes keep two key ideas:

- it is common for very large training setups
- idle time between pipeline stages appears as the familiar **pipeline bubble**

## Communication collectives

### All-reduce

All-reduce aggregates data across devices and leaves the final full result on every device.

The source repeatedly uses it for:

- data-parallel gradient synchronization
- row-parallel tensor-parallel outputs that must be summed

Main trade-off preserved from the notes:

- communication volume is high
- every device ends up holding the full reduced result

### Reduce-scatter

Reduce-scatter combines reduction with sharding.

The notes emphasize why it is useful for FSDP and ZeRO:

- each device keeps only the shard it is responsible for
- this lowers both memory pressure and communication overhead relative to keeping a full replicated result

### All-gather

All-gather takes sharded data and reconstructs a full copy on every device.

The clearest training example from the source is FSDP forward execution:

- weights are stored as shards
- when a layer is executed, devices all-gather the full weight
- computation runs with the reconstructed full weight

### Relationship between the collectives

The training notes keep the standard identity:

- `Reduce-Scatter + All-Gather = All-Reduce`

## Column-parallel and row-parallel linear layers

The training sub-page includes a compact version of tensor-parallel linear algebra.

### Column-parallel linear

For `Y = XW`, split `W` by output columns.

- each device computes part of the output channels
- this naturally produces output shards
- depending on the next layer, those shards can often remain distributed for a while

### Row-parallel linear

For `Y = XW`, split `W` by input rows and split `X` the same way.

- each device computes a full-shape partial contribution
- the final output requires summing those partial results
- this is where all-reduce appears

## Sharded-array matmul cases

The source also contains useful notes on how communication depends on whether the contracting dimension is sharded.

### Case 1: the contracting dimension is not sharded

Example pattern:

- `A[I_X, J] * B[J, K_Y] -> C[I_X, K_Y]`

If the contracting dimension stays local, each device can do local matmul and reduction without extra communication.

### Case 2: one multiplicand is sharded on the contracting dimension

Example pattern:

- `A[I, J_X] * B[J, K] -> C[I, K]`

Because the full contracting dimension is needed for the matmul, the source notes that an **all-gather** is typically needed first to reconstruct the full `J` dimension.

### Case 3: both multiplicands are sharded on the same contracting dimension

Example pattern:

- `A[I, J_X] * B[J_X, K] -> C[I, K]`

Local products are valid, but each device only computes a **partial sum** of the final result. To finish the operation, the source notes that an **all-reduce** is needed along that mesh axis.

## Conservative review notes

These notes remain intentionally close to the source and should still be reviewed if you want a more formal final version:

- the source briefly mentions Megatron parallelism and sequence parallelism without developing them
- the communication formulas are more conceptual than exhaustive
- some terminology mixes TPU-oriented and GPU-oriented language, which has been normalized only lightly here
