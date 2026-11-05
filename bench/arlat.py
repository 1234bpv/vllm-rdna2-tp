"""All-reduce latency at decode-sized tensors (hidden size 5120, fp16).

Run one process per GPU, for example:
    torchrun --nproc-per-node 2 arlat.py
"""
import time

import torch
import torch.distributed as dist


def bench(n, iters=300):
    x = torch.ones(n, dtype=torch.float16, device="cuda")
    for _ in range(30):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
res = [(n, bench(n)) for n in (5120, 10240, 40960)]
if rank == 0:
    print("  " + "  ".join(f"{n}:{us:7.1f} us" for n, us in res), flush=True)
dist.destroy_process_group()
