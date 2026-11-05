# Related work

Other public projects that run LLMs on gfx1030 or that exchange data through host memory. This list comes from a search in November 2026, and it may be incomplete.

**The numbers are self-reported by each project and were not reproduced here.** Conditions differ. In particular, the PRO V620 has the same chip as the tested cards, but 32 GB of memory and 72 CUs, while the RX 6800 has 16 GB and 60 CUs.

| Project | What it is | Kernels without matrix units | TP without P2P through the host | Reported results |
|---|---|---|---|---|
| [opengfx1030/vllm-rdna](https://github.com/opengfx1030/vllm-rdna) | Fork of vLLM v0.31.0 for the PRO V620. The closest project | Yes: W4A16 through `v_dot2_f32_f16`, optional W4A8 through `sdot4` | No: its push all-reduce requires P2P. With `P2P=0` it uses the stock RCCL path | With MTP, one stream: 81.74 and 96.45 tokens/s per request (model not named in the part that was checked). In [a vLLM issue comment](https://github.com/vllm-project/vllm/issues/38107): a model named only "27B", INT4 weights, TP=2, MTP, on Radeon PRO V620: 45.0 tokens/s |
| [exaroot/gfx1030-vllm-rdna2-extras](https://hub.docker.com/r/exaroot/gfx1030-vllm-rdna2-extras) | Docker image of vLLM for gfx1030 | Yes: W4A16 | No: usual TP=2 | Qwen3.8-27B AWQ, 2 × V620: TP=1 17.2, TP=2 28.5 tokens/s |
| [darkkernel/vllm-gfx1030](https://hub.docker.com/r/darkkernel/vllm-gfx1030) | Docker image of vLLM 0.19.1 for gfx1030 | No | No | None |
| [sebastianmechno-sys/vllm-rocm-windows-rdna2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2) | vLLM on Windows, RX 6750 XT (gfx1031) | W4 GEMV, technique not checked | No | 4B model: 59.4 tokens/s |
| [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) | llama.cpp patches, gfx1030 described as best effort | K-quant kernels | No: its host-staged block targets RDNA4 | [Qwen3.8-27B Q6_K_XL, V620](https://github.com/stew675/llama-cpp-rdna-boosts/pull/165): pp1024 478.1, tg128 18.12 tokens/s |
| [tuandat3019/rdna2-llamacpp-optimizations](https://github.com/tuandat3019/rdna2-llamacpp-optimizations) | llama.cpp on Windows, RX 6800 + RX 6600 | No | Reports TP without P2P on Windows as unusable, TG 4.11 tokens/s | RX 6800, Qwen3.8-27B IQ4_XS: 117.07 tokens/s at 8K, with MTP |
| [Kmic-68/llama.cpp, PR 12](https://github.com/Kmic-68/llama.cpp/pull/12) | The closest method: exchange through a host flag without P2P, but on NVIDIA P100 with CUDA | | Yes, on other hardware and another engine | pp512 +20 %, tg256 +2 % |
| [warpfront/hipfire](https://github.com/warpfront/hipfire) | Own engine in Rust. RDNA2 marked as unvalidated | | No | RDNA3 and RDNA4 only |

**Papers.** [CoMoE, arXiv:2610.09424](https://arxiv.org/abs/2610.09424) routes the token exchanges of MoE expert parallelism through host memory on 8 × RTX 5090 without P2P. This is not the dense all-reduce of tensor parallelism. No papers about LLMs on RDNA2 were found on arXiv or through web search of ACM and IEEE; dblp and the ACM and IEEE sites could not be searched directly.

**Upstream.** [vLLM PR 52391](https://github.com/vllm-project/vllm/pull/52391) proposed RDNA2 work and was closed by its author without merge and without a review by the code owners, so it says nothing about the view of vLLM on this target.

**ROCm.** AMD lists gfx1030 only as W6800 and V620.

## What is different here

The combination of tensor parallelism in vLLM with an own all-reduce through host memory on cards without P2P. If your cards have P2P access, for example PRO V620 in one server, look at `opengfx1030/vllm-rdna` first.

No comparison with these projects was measured, so no claim is made about which one is faster or who did what first.
