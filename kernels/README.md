# Kernels for gfx1030

Self-contained prototypes. Each one builds with a single command and checks its own
result against direct computation on 12 random elements. Without that check none of
the numbers below can be trusted: twice during development a "fast" kernel turned out
fast only because it computed the wrong thing.

```bash
hipcc --offload-arch=gfx1030 -O3 -Wno-unused-value KERNEL.hip -o KERNEL && ./KERNEL
```

| File | What it is | TOPS |
|---|---|---:|
| `dot-peak.hip` | instruction ceiling: fp16 / int8 / int4 | 23.2 / 46.5 / 92.6 |
| `int4-gemm.hip` | W4A4 sandbox: variants from 9.6 to 52.1 | 52.1 |
| `int4-prod.hip` | W4A4 in the production format, scale per row | 50.5 |
| `int4-prodg1.hip` | the same, but scale per GROUP and a single sA | 49.3 |
| `int8-prods.hip` | **W4A8 with signed activations**, the variant that went into production | **25.3** |
| `quant-act.hip` | activation quantizer to 4 bits | 0.18 ms |
| `quant-act8.hip` | activation quantizer to 8 bits | 0.11 ms |
| `occ-sweep.hip` | occupancy and LDS layout variant sweep | — |

For comparison, `torch.mm` in fp16 through hipBLASLt on the same shape
(M=2048, K=5120, N=17408) gives **20.2 TOPS**.

⚠️ The numbers were measured on an RX 6800 at 1600 MHz. The ceiling is computed as
clock × compute units × lanes × operations, that is, 1600 MHz × 60 CU × 64 × N.
