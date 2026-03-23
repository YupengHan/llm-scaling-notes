# Source Manifest

This pass refined the existing repo in place using the full set of Notion export archives. All work stayed local. No remote was created, nothing was pushed, and the original archives were not modified.

## Archive handling

| Source archive name | Outer archive unpacked | Nested `Part-1.zip` unpacked | Notes |
| --- | --- | --- | --- |
| `jax-scaling-book.zip` | yes | yes | Root study tree export |
| `jax-scaling-book_LLM-System-Design.zip` | yes | yes | Sub-page export referenced by the root notes |
| `jax-scaling-book_Training-how-to-scael.zip` | yes | yes | Sub-page export referenced by the root notes |
| `jax-scaling-book_llma2CPP.zip` | yes | yes | Sub-page export referenced by the root notes |
| `jax-scaling-book_Inference.zip` | yes | yes | Sub-page export referenced by the root notes |

## Unified source content manifest

| Source archive name | Source markdown file name | Inferred source title | Status | Mapped output file | Brief note about what was added or changed |
| --- | --- | --- | --- | --- | --- |
| `jax-scaling-book.zip` | `JAX Scaling Book 319bb8eb37cc808a94cbc2006c006ca6.md` | JAX Scaling Book | processed | `README.md`; `docs/overview.md`; `docs/roofline.md`; `docs/transformer-systems.md`; `docs/tpu-systems.md`; `docs/communication.md`; `docs/tensor-parallelism.md`; `resume/experience-bullets.md` | Preserved the existing root-note structure and used it as the base for the repo. Updated the repo framing so the root page is no longer treated as the only meaningful source. |
| `jax-scaling-book.zip` | `Screenshot_from_2026-03-04_13-56-43.png` | Roofline diagram | processed | `assets/images/roofline-overview.png` | Kept because it materially supports the roofline notes. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-04_at_22.45.12.png` | TPU architecture screenshot | processed | `assets/images/tpu-architecture.png` | Kept because it helps the TPU architecture section. |
| `jax-scaling-book.zip` | `Screenshot_from_2026-03-05_17-50-30.png` | TPU pod and network screenshot | processed | `assets/images/tpu-network-topology.png` | Kept because it supports the TPU networking notes. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-07_at_15.51.15_2.png` | Contracting and batching diagram | processed | `assets/images/matmul-contracting-and-batching.png` | Kept because it anchors the FLOPs explanation. |
| `jax-scaling-book.zip` | `image.png` | Transformer dimension graph | processed | `assets/images/transformer-dimension-graph.png` | Kept because it supports the transformer cost section. |
| `jax-scaling-book.zip` | `image 2.png` | Mesh sharding example 1 | processed | `assets/images/mesh-sharding-example-1.png` | Kept because it clarifies mesh-based sharding. |
| `jax-scaling-book.zip` | `image 3.png` | Mesh sharding example 2 | processed | `assets/images/mesh-sharding-example-2.png` | Kept because it clarifies mesh-based sharding. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-07_at_22.24.42.png` | Attention module screenshot | skipped | — | The underlying content was preserved in text, so the screenshot itself was not required. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-07_at_22.26.18.png` | MLP module screenshot | skipped | — | The underlying content was preserved in text, so the screenshot itself was not required. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-07_at_22.49.05.png` | Other FLOPs screenshot | skipped | — | The main point was preserved in text, so the image was not essential. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-08_at_01.31.44.png` | Transformer takeaway screenshot 1 | skipped | — | Converted into cleaned text instead of preserving the screenshot. |
| `jax-scaling-book.zip` | `Screenshot_2026-03-08_at_03.02.46.png` | Transformer takeaway screenshot 2 | skipped | — | Converted into cleaned text instead of preserving the screenshot. |
| `jax-scaling-book.zip` | `image 1.png` | MoE diagram | unclear | — | The text was strong enough to preserve the concept without the image, but the diagram could still be added in a later pass if you want a more visual MoE section. |
| `jax-scaling-book_LLM-System-Design.zip` | `LLM System Design 323bb8eb37cc8000b3e7cf6779b8e1c1.md` | LLM System Design | processed | `docs/llm-serving-system-design.md`; `docs/tensor-parallelism.md`; `docs/communication.md`; `README.md`; `resume/experience-bullets.md` | Added the previously missed serving/system-design material, including TP/DP/PP trade-offs, KV cache, paged KV cache, continuous batching, and chunked prefill. Also used it to sharpen the communication and tensor-parallelism docs. |
| `jax-scaling-book_LLM-System-Design.zip` | `image.png` | GQA and tensor-parallel diagram | skipped | — | The diagram was helpful but not necessary once the shape explanations were preserved in text. |
| `jax-scaling-book_Training-how-to-scael.zip` | `Training how to scale 325bb8eb37cc80d099f5c670f0c5c07c.md` | Training how to scale | processed | `docs/training-scaling.md`; `docs/communication.md`; `docs/tensor-parallelism.md`; `README.md`; `resume/experience-bullets.md` | Added the previously missed training-focused notes on DP, FSDP/ZeRO, tensor parallelism, pipeline parallelism, collectives, and sharded-array matmul cases. |
| `jax-scaling-book_Training-how-to-scael.zip` | `image.png` | Data parallelism diagram | skipped | — | The concept is straightforward in text and did not need the image to remain clear. |
| `jax-scaling-book_Training-how-to-scael.zip` | `image 1.png` | FSDP/ZeRO diagram | skipped | — | The textual explanation was sufficient for this pass. |
| `jax-scaling-book_Training-how-to-scael.zip` | `image 2.png` | Tensor parallelism diagram | skipped | — | The main logic was integrated into text-based docs. |
| `jax-scaling-book_llma2CPP.zip` | `Llama2 cpp 322bb8eb37cc8001957dcd3f68877e5f.md` | Llama2.cpp | processed | `docs/llama2-cpp.md`; `README.md`; `resume/experience-bullets.md` | Added the previously missed implementation-oriented notes on config interpretation, RoPE, and weight tying. |
| `jax-scaling-book_Inference.zip` | `Inference 321bb8eb37cc80499da5c8506e66a585.md` | Inference | processed | `docs/inference-systems.md`; `README.md`; `docs/overview.md` | Added a dedicated inference doc covering prefill vs. decode, KV cache complexity, inference bottlenecks, distribution strategy, and serving-engine design patterns. |
| `jax-scaling-book_Inference.zip` | `image.png` | Prefill and generation tensor-flow diagram | skipped | — | The tensor-shape content was preserved directly in text, and this image duplicates the root export's `image.png`, so keeping another copy was unnecessary. |
| `jax-scaling-book_Inference.zip` | `Screenshot_from_2026-03-12_18-43-01.png` | Decoding latency formula screenshot | skipped | — | The latency heuristic was rewritten directly in Markdown, so the screenshot was not necessary. |
| `jax-scaling-book_Inference.zip` | `Screenshot_from_2026-03-12_18-43-28.png` | Decoding latency trade-off screenshot | skipped | — | The small-batch vs. large-batch trade-off was preserved in text. |
| `jax-scaling-book_Inference.zip` | `image 1.png` | Tensor-parallel inference diagram | skipped | — | The TP plus communication flow was preserved in text without keeping the diagram, and this image duplicates `image 2.png` from the training export. |
| `jax-scaling-book_Inference.zip` | `image 2.png` | Batched prefill-then-generate diagram | skipped | — | The serving pattern was documented in prose without requiring the image. |
| `jax-scaling-book_Inference.zip` | `image 3.png` | Interleaved prefill and decode diagram | skipped | — | The scheduling pattern was documented in prose without requiring the image. |
| `jax-scaling-book_Inference.zip` | `image 4.png` | Disaggregated inference diagram | skipped | — | The disaggregated serving design was preserved in text without keeping the image. |
| `jax-scaling-book_llma2CPP.zip` | `Deep_Learning-47.jpeg` | RoPE illustration | skipped | — | The mathematical notes were kept in text and the image was not strictly necessary. |
| `jax-scaling-book_llma2CPP.zip` | `image.png` | Config diagram | skipped | — | The config fields were clearer when rewritten directly into Markdown. |

## Skipped or unclear items

- `Part_7 Inference` in the root export is now backed by the dedicated `Inference` sub-page export, but some subsections still remain outline-like or unfinished.
- `Part_12 GPUs` is still an outline with only light detail in the source.
- The numeric placeholders `6`, `8`, and `11` remain unclear in the root export.
- The `MoE` image from the root export could still be added later if you want a more visual section.
- The system-design file contains some duplicated content blocks; this pass kept the strongest cleaned version rather than repeating all duplicates.

## Main gaps fixed from the previous pass

1. The repo now inspects and incorporates **all five archives**, not just the root archive.
2. The four sub-page exports are now represented as actual docs rather than just unresolved pointers.
3. The source manifest now tracks the complete source set and clearly records processed, skipped, and unclear items.
4. The README now reflects the full study tree rather than implying the repo came from one long root-page outline only.
5. The inference section now has a dedicated doc instead of remaining mostly a placeholder in the root notes.
