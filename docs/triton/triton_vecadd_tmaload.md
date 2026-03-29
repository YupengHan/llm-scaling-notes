# VecAdd TmaLoad

## 1. Three level abstraction

1. **Python host-side code** launches the kernel and chooses the launch grid / meta-parameters.
2. **The Triton DSL** (`triton.language as tl`) describes tensor and pointer operations inside the kernel.
3. **The compiler and backend** lower the kernel to GPU-specific code.

## 2. Core execution concepts

- `tl.program_id(axis=0/1/2)` returns the program instance id along one(x/y/z) axis of Triton's launch grid.
- Triton program instance = CUDA **thread block / CTA**. Even if Cluster/CGA is introduced, program_id still maps to only a single Block.
- A Triton launch grid can be a tuple or a callable that returns a tuple.
- When a `torch.Tensor` is passed to a Triton kernel, Triton treats it as a pointer to the tensor's first element.

## 3 GPU Mem Arch

| 项目     | H100                                | B200                              |
| -------- | ----------------------------------- | --------------------------------- |
| SRAM     | ~10–20+ TB/s, L2=50MB,smem=228KB/SM | ~20+ TB/s , L2=126MB,smem=KB / SM |
| HBM      | 80/94 GB, 3.35/3.9 TB/s             | 180GB, ~8 TBs                     |
| CPU DRAM | ~TB, NVLink 900 GB/s                | ~TB, NVLink 900 GB/s              |

## 4. Simple kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def write_pid_kernel(output_ptr):
    pid = tl.program_id(axis=0) # blockIdx.x, axis=1 -> blockIdx.y
    # program instance = CUDA thread block / CTA
    tl.store(output_ptr + pid, pid)

output = torch.empty(4, dtype=torch.int32, device="cuda")
write_pid_kernel[(4,)](output) # (4,) is grid size, grid(gx,gy) -> gridDim = (gx,gy)
print(output.cpu())
```

This launches four program instances(block). Each instance(block with size 1) writes its own `pid` into one output element.

## 5. Masked vector-add kernel

```python
import torch
import triton
import triton.language as tl

DEVICE = "cuda"
BLOCK_SIZE = 1024


@triton.jit
def add_mask_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add_mask(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_mask_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### Explanation

- Each program instance handles one contiguous block of elements.
- `offsets` computes the global indices for that block.
- `mask` prevents out-of-bounds loads and stores in the tail block.
- The kernel is memory-bound, so the main concern is clean global-memory access rather than complicated arithmetic.
- `cdiv` means ceiling division, triton.cdiv(a, b) computes the number of blocks needed to cover all elements by dividing a by b and rounding up.

## 6. TMA load vector add

Uses `tl.make_block_ptr`, which makes the memory region handled by each program instance more explicit.

```python
@triton.jit
def add_block_ptr_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    x = tl.load(x_block_ptr, boundary_check=(0,))
    y = tl.load(y_block_ptr, boundary_check=(0,))
    z = x + y
    tl.store(z_block_ptr, z, boundary_check=(0,))


def add_block_ptr(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_block_ptr_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### Triton tl.make_block_ptr

Brief, cheat-sheet style breakdown of the `tl.make_block_ptr` parameters:

### `tl.make_block_ptr` Quick Reference

```python
x_block_ptr = tl.make_block_ptr(
    base=x_ptr,           # 1. starter ptr
    shape=(M, N),         # 2. data shape
    strides=(stride_m, 1),# 3. mem stride
    offsets=(m_off, n_off),# 4. curr block starting index
    block_shape=(BM, BN), # 5. current block shape
    order=(1, 0)          # 6. mem dim order
)
```

**Parameter Breakdown:**

- **`base`**: The global memory address of the starting element (usually passed in from Python/PyTorch).
- **`shape`**: The full, original dimensions of the entire tensor in global memory.
- **`strides`**: The physical memory jump (in elements) required to move exactly one step along each dimension.
- **`offsets`**: The starting index (coordinates) of the specific chunk/block this thread block is responsible for loading.
- **`block_shape`**: The dimensions of the tile/block you are bringing into SRAM.
- **`order`**: Defines memory contiguity, ordered from **fastest-changing (most contiguous) to slowest**.
  - _Row-major (PyTorch default):_ `(1, 0)` -> Dim 1 (columns) is physically contiguous.
  - _Column-major:_ `(0, 1)` -> Dim 0 (rows) is physically contiguous.
  - _1D Tensor:_ `(0,)`

## 7. Benchmarking workflow

- perf_report(Benchmark(...))
  - triton.testing.Benchmark is a configuration object, not something that executes the benchmark. It defines the experimental setup and attaches a “test configuration” to the benchmark(size, provider) function.
  - triton.testing.perf_report(...) is a decorator that binds the Benchmark(...) configuration to the benchmark function.
    `pythonbenchmark = perf_report(config)(benchmark)`
- triton.testing.do_bench(...)
  - Does NOT connect to perf_report.
  - It is a low-level timing utility / benchmark runner.
  - It repeatedly executes a given lambda function and measures execution time.

```python
import torch
import triton
import triton.language as tl

DEVICE = "cuda"
BLOCK_SIZE = 1024


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(20, 27)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton_mask", "triton_block_ptr"],
        line_names=["PyTorch", "Triton (Mask)", "Triton (Block Ptr)"],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == "triton_mask":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_mask(x, y), quantiles=quantiles)
    elif provider == "triton_block_ptr":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_block_ptr(x, y), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this example.")

    test_size = 10_000_000
    x_test = torch.randn(test_size, device=DEVICE)
    y_test = torch.randn(test_size, device=DEVICE)

    torch.testing.assert_close(add_mask(x_test, y_test), x_test + y_test)
    torch.testing.assert_close(add_block_ptr(x_test, y_test), x_test + y_test)
    print("Correctness checks passed.")

    benchmark.run(save_path=".", print_data=True, show_plots=False)
```

## References

- [Triton Tutorial](https://github.com/evintunador/triton_docs_tutorials)
