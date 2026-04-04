# High-Throughput LLM Inference Systems

## Overview

This note covers the core internals of a production-grade LLM inference engine (based on vLLM architecture), organized into the following topics:

1. **LLM Engine** — scheduling, paged attention, continuous batching
2. **Advanced Features** — chunked prefill, prefix caching, guided & speculative decoding, disaggregated prefill/decode
3. **Scalability** — single GPU → multi-GPU
4. **Benchmarks & Auto-tuning** — throughput and latency metrics

---

## ToDo

- `EngineCoreRequest: cache_salt`
- `SOM — Structured Output Manager`
- `KV Cache Manager`
- `DeepSeek-V2 MLA`
- `KV Cache C++ Design`

---

## LLM Engine Architecture

![vLLM Engine Architecture](../../assets/images/vllm-llm-engine.png)

The engine is structured as a layered pipeline:

```
requests in → Processor → [Engine Core: Scheduler + Model Executor] → Output Processor → result out
```

The **engine core client** acts as the external interface, while the **engine core** contains the actual scheduling and execution logic. This separation enables distributed deployments where a client proxy dispatches work to remote engine cores via RPC.

---

### Processor

Responsible for **input tokenization, validation**, and packaging requests into `EngineCoreRequest` objects. On the output side, it handles **detokenization** and converts `EngineCoreOutputs` back into user-visible `RequestOutput`.

### Tokenization: Two Tables

| Table            | Content                                         | Location |
| ---------------- | ----------------------------------------------- | -------- |
| Tokenizer vocab  | string → token_id mappings + tokenization rules | CPU      |
| Embedding matrix | token_id → embedding vector                     | GPU HBM  |

After execution, the output side performs detokenization and converts internal engine outputs into a user-visible response object.

### `EngineCoreRequest` Structure

`EngineCoreRequest` is a **request-level** structure, not a token-level one. It packages the prompt, generation parameters, and scheduling metadata needed by the engine.

```cpp
EngineCoreRequest = {
  request_id,         // unique request identifier
  prompt_token_ids,   // tokenized prompt
  sampling_params,    // temperature / top_p / max_tokens / ...
  arrival_time,       // request arrival timestamp
  priority,           // optional scheduling priority
  data_parallel_rank, // target DP worker if applicable
  // optional: LoRA, prompt adapters, multimodal payloads, cache metadata, ...
}
```

#### Key fields

| Field                | Meaning                                      | Why it matters                                                                       |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------ |
| `request_id`         | Unique identifier for one request            | Used for scheduler bookkeeping, KV-cache ownership, and streaming output routing     |
| `arrival_time`       | Timestamp when the request enters the engine | Needed for FCFS/fairness policies and latency accounting                             |
| `priority`           | Optional priority level                      | Allows the scheduler to prioritize urgent requests                                   |
| `sampling_params`    | Generation policy                            | Controls decoding behavior such as temperature, top-p, and max tokens                |
| `data_parallel_rank` | Assigned DP worker                           | Maps the request to a worker in distributed execution                                |
| `cache_salt`         | Optional cache-isolation key                 | Prevents unintended prefix-cache sharing across requests that should remain isolated |

---

### Scheduler

Decides **which requests to execute in the next engine step**.

- **Scheduling policies**: FCFS (first-come, first-served) or priority-based
- **Request queues**: maintains a `waiting` queue and a `running` queue
- Closely coupled with the **KV Cache Manager** — scheduling decisions are gated by available KV cache blocks

### KV Cache Manager

The KV Cache Manager abstracts GPU memory into a **paged block system**, inspired by virtual memory paging.

#### Block Definition

A single block covers one layer and a contiguous token range:

```
block = (layer l, tokens [i ... i + block_size - 1])
```

Each block stores the key and value tensors for that layer and token range.

#### Capacity Calculation

- Blocks per layer per sequence: `ceil(seq_len / block_size)`
- Total blocks per sequence: `num_layers × ceil(seq_len / block_size)`

**Example**:

- `num_layers = 80`
- `block_size = 16`
- `seq_len = 35`

Then:

- blocks per layer = `ceil(35 / 16) = 3`
- total blocks = `80 * 3 = 240`

#### Block Size Formula (Standard Transformer, non-MLA)

```
bytes_per_block = 2 (K+V) × block_size × num_kv_heads × head_size × dtype_bytes
```

For bf16: `dtype_bytes = 2`

---

#### Indexing Structure vs. Actual KV Memory

The design separates **control plane (CPU)** from **data plane (GPU)**:

| Layer                  | Location | Contents                                                                                   |
| ---------------------- | -------- | ------------------------------------------------------------------------------------------ |
| **Indexing structure** | CPU      | `free_block_queue`, `block_pool` metadata, per-sequence block tables, token→block mappings |
| **Actual KV memory**   | GPU HBM  | Real K tensors and V tensors, stored in paged block layout                                 |

The `free_block_queue` holds `block_id` integers — not actual KV data. This is a lightweight index that enables O(1) block allocation and deallocation.

#### Runtime Behavior

1. **Initialization**: pre-allocate a large contiguous HBM region, partition into fixed-size blocks
2. **New request / prefill**: allocate blocks proportional to prompt length
3. **Decode step**: if the last block is full, allocate one new block; otherwise reuse it
4. **Request completion**: return all blocks to the free queue

#### Why Paged Blocks vs. Contiguous Allocation

| Contiguous allocation                    | Paged blocks                             |
| ---------------------------------------- | ---------------------------------------- |
| Must pre-reserve max possible length     | Grows on demand                          |
| Severe fragmentation                     | Minimal fragmentation                    |
| Hard to batch requests of varied lengths | Scheduler can freely mix request lengths |
| Simple address arithmetic                | Requires block table indirection         |

---

## Model Executor

Drives the **forward pass** of the model.

- Receives a scheduled batch from the scheduler: token IDs + block tables
- Prepares the corresponding tensors
- Invokes CUDA kernels (attention, feed-forward, etc.)
- The block table is passed to the attention kernel so it can locate the correct KV blocks in GPU memory

---

### SOM — Structured Output Manager

`SOM` (Structured Output Manager) is integrated into the engine core to support **guided decoding** — constraining model outputs to conform to a specified schema (e.g., JSON, regex, context-free grammar). Token sampling is masked at each step to enforce structural validity.

---

### Engine Core Client & Engine Core

The engine is split into two layers:

| Layer                  | Role                                                                  |
| ---------------------- | --------------------------------------------------------------------- |
| **Engine Core Client** | External-facing interface; acts as a business-logic proxy             |
| **Engine Core**        | Internal host-side management logic (scheduling, KV cache, execution) |

This separation becomes critical for **large-scale deployments** requiring Tensor Parallelism (TP) or Pipeline Parallelism (PP) across multiple nodes. The client acts as a front-end proxy, forwarding requests to remote Engine Core instances via RPC.

---

### RPC in LLM Inference

#### Why RPC Is Needed

- **Cross-node orchestration**: a master node receives user requests and dispatches tasks to worker nodes
- **Disaggregated serving**: when prefill and decode are separated onto different machine groups, RPC coordinates task handoff and state transfer

#### Control Plane vs. Data Plane

| Plane             | Protocol         | Content                                                                                  | Characteristics                  |
| ----------------- | ---------------- | ---------------------------------------------------------------------------------------- | -------------------------------- |
| **Control plane** | RPC (gRPC / Ray) | Request metadata: request ID, token count, generation params, KV cache block assignments | Small payload, ultra-low latency |
| **Data plane**    | NCCL             | Tensor data: TP all-reduce, cross-node KV cache / activation transfer                    | High bandwidth, GPU-direct       |

#### RPC Performance Considerations

Key sources of overhead in RPC:

1. **Serialization / deserialization** — encoding and decoding message payloads
2. **Network transport** — kernel/user-space context switch overhead
3. **Copy count** — number of memory copies along the send/receive path

Minimizing these is critical for keeping control-plane latency below the compute time of a single decode step.
