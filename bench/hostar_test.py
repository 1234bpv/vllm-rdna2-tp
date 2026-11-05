"""Check host_all_reduce across two processes: correctness and latency.

Needs a vLLM build with the ROCm custom ops (torch.ops._rocm_C.host_ar_*).
Run one process per GPU, for example:
    torchrun --nproc-per-node 2 hostar_test.py
"""
import os
import time

import torch
import torch.distributed as dist
import vllm._custom_ops  # noqa: F401  registers torch.ops._rocm_C

r = int(os.environ["RANK"])
w = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(r)
dev = f"cuda:{r}"
dist.init_process_group("nccl", rank=r, world_size=w)
ops = torch.ops._rocm_C
ops.host_ar_init(r, f"/vllm_host_ar_test_{os.getuid()}")

bad = 0
checks = 0
for it in range(1, 51):
    for n in (5120, 10240, 40960):
        x = torch.full((n,), float((r + 1) * it), dtype=torch.float16, device=dev)
        out = torch.empty_like(x)
        ops.host_all_reduce(x, out)
        torch.cuda.synchronize()
        want = torch.full((n,), float(3 * it), dtype=torch.float16, device=dev)
        checks += 1
        if not torch.equal(out, want):
            bad += 1
            if bad == 1:
                print(f"[rank {r}] first mismatch n={n} it={it}: {out[:4].tolist()}, expected {3 * it}",
                      flush=True)


def bench(n, iters=300):
    x = torch.ones(n, dtype=torch.float16, device=dev)
    out = torch.empty_like(x)
    for _ in range(30):
        ops.host_all_reduce(x, out)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.host_all_reduce(x, out)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


lat = [(n, bench(n)) for n in (5120, 10240, 40960)]
print(f"[rank {r}] errors {bad}/{checks} | " +
      "  ".join(f"{n}:{us:6.1f} us" for n, us in lat), flush=True)
ops.host_ar_destroy()
dist.destroy_process_group()
