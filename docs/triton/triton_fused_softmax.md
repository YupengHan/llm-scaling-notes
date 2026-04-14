# Fused Softmax in Triton

This note is a cleaned-up walkthrough of Triton fused softmax. The main goal is to explain **why the fused version is faster**, **how the launch wrapper decides the grid**, and **what `num_warps` / `num_stages` are doing**.

## 1. Why fuse softmax?

### Naive but numerically safe softmax

```python
def naive_softmax(x):
    """
    Main issue: too many HBM / DRAM reads and writes.
    The fused Triton version keeps each row on chip (registers / shared memory)
    instead of repeatedly materializing intermediate tensors in global memory.
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    out = numerator / denominator[:, None]

    # in total we do 8MN + 4M element-wise memory operations
    # (read 5MN + 2M elements; write 3MN + 2M elements)
    return out
```

The bottleneck here is memory traffic, not arithmetic. The fused kernel tries to read each row once, do the reduction and normalization on chip, and write the result once.

## 2. Launch-side logic

### GPU properties used by the launcher

```python
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]          # total register budget per SM, in 32-bit registers
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]  # shared memory capacity per SM
WARP_SIZE = properties["warpSize"]             # usually 32 on NVIDIA, 64 on AMD
```

### Wrapper that prepares the kernel launch

```python
def softmax(x):
    """
    Launch helper: prepares meta-parameters and decides how many persistent
    programs to launch.
    """
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # Number of elements handled in one row-wide block, not blockDim.x.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # blockDim.x = num_warps * 32 on NVIDIA.
    # This is a simple manual heuristic, not a universal rule.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    y = torch.empty_like(x)

    kernel = _softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()

    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    # CUDA-side occupancy intuition.
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)

    grid = (num_programs, 1, 1)
    kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )
    return y
```

### What happens in `warmup()`?

`warmup()` does not launch the real workload. It triggers JIT compilation and specialization, then returns a `CompiledKernel` object so the runtime can inspect metadata such as register usage (`n_regs`) and shared-memory usage (`metadata.shared`).

### What happens in `_init_handles()`?

The compiled binary is lazily initialized. `_init_handles()` is the step that actually prepares runtime handles. Conceptually, it does three things:

1. creates the launcher,
2. checks whether the shared-memory requirement fits on the current device,
3. loads the compiled binary onto the device and exposes runtime resource info such as:
   - `module`
   - `function`
   - `n_regs`
   - `n_spills`
   - `n_max_threads`

### Overall launch flow

1. Compile once and inspect resource usage.
2. Estimate occupancy from registers and shared memory.
3. Launch a persistent set of programs instead of one program per row.

## 3. Kernel walkthrough

```python
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,    # number of elements to skip when moving to the next row
    n_rows,
    n_cols,               # matrix dimensions
    BLOCK_SIZE: tl.constexpr,   # smallest power of two >= n_cols
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)      # first row assigned to this program
    row_step = tl.num_programs(0)     # total number of persistent programs

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Move to the beginning of row row_idx.
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # For a contiguous [M, N] tensor in row-major layout, input_row_stride is
        # usually N rather than 1. Passing stride explicitly also makes the kernel
        # work for non-contiguous views.

        # Load one row into on-chip memory. BLOCK_SIZE is >= n_cols, so we need masking.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf'))

        # Numerical stability: subtract the row max before exponentiation.
        # Masked-out lanes stay at -inf and therefore do not affect softmax.
        row_minus_max = row - tl.max(row, axis=0)

        # Triton's exp is fast but approximate.
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # Store only the valid columns.
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

## 4. Key concepts behind the kernel

### Triton program vs. CUDA block

A Triton **program** can be thought of as a program instance that **roughly maps to a CUDA thread block / CTA** at launch time. `num_warps` controls how many warps cooperatively execute one program.

```cpp
threads_per_program = num_warps * 32;   // NVIDIA
```

### Why `stride(0)` matters

`input_row_stride` is the distance, in elements, between two consecutive rows. For a contiguous `[M, N]` tensor, it is usually `N`, not `1`. This is important because the kernel should also work when the input is a non-contiguous view.

### What `num_stages` really controls

`num_stages` here controls **software pipelining inside the loop of one program**. It is different from ordinary SM-level latency hiding caused by having multiple thread blocks resident on the same SM.

SM-level scheduling looks like this:

- block A is waiting on global memory,
- block B is doing compute,
- the SM overlaps them to hide latency.

`num_stages` is different. It pipelines **different loop iterations inside one Triton program**.

Without software pipelining, the loop is closer to:

```text
iter 0: load row0 -> compute softmax row0 -> store row0
iter 1: load row1 -> compute softmax row1 -> store row1
iter 2: load row2 -> compute softmax row2 -> store row2
```

With `num_stages = 2`, the compiler can overlap neighboring iterations more like:

```text
stage A:  load row0
stage B:  compute/store row0    +    load row1
stage C:  compute/store row1    +    load row2
stage D:  compute/store row2    +    load row3
```

That is the simplest form of double buffering / software pipelining: while the program is computing the current row, it can already start preparing data for the next row.

### Trade-offs of larger `num_stages`

Benefits:

- stronger latency hiding,
- more overlap between iterations,
- sometimes better throughput.

Costs:

- higher register pressure,
- longer live ranges for shared-memory / intermediate values,
- potentially lower occupancy,
- more complex compiler scheduling.

## 5. Appendix: small syntax reminders

### `torch.max(..., dim=1)[0]`

```python
x_max = x.max(dim=1)[0]  # shape [M] when x has shape [M, N]
# max returns:
# torch.return_types.max(values=..., indices=...)
```

### `[:, None]`

```python
# x: [M, N], x_max: [M]
# x_max[:, None] is equivalent to x_max.reshape(M, 1)
z = x - x_max[:, None]   # shape [M, N]
```

This inserts a new axis at dimension 1 so broadcasting works row-wise.

### Broadcasting reminder

NumPy / PyTorch broadcasting aligns dimensions from right to left, pads missing dimensions with 1, and then expands where allowed.

## Reference

This note is based primarily on Triton's official fused softmax tutorial and on Triton runtime/compiler source for the `warmup()` and `_init_handles()` details.
