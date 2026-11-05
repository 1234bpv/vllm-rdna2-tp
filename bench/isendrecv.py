"""isend/irecv between two ranks: the path pipeline parallelism uses to pass activations between stages.

Blocking send/recv was reliable on the test machine (an earlier run recorded 3 runs of
100 checks, 0 errors). vLLM sends with isend_tensor_dict, which is asynchronous and
takes a different path through RCCL, so this script checks that path directly.

Run one process per GPU, for example:
    torchrun --nproc-per-node 2 isendrecv.py
"""
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"
dist.init_process_group("nccl", rank=rank, world_size=world)
bad = 0
checks = 0
for it in range(50):
    for n in (1280, 5120, 65536):
        checks += 1
        if rank == 0:
            t = torch.full((n,), float(it + 1), device=dev)
            h = dist.isend(t, 1)
            h.wait()
            t2 = torch.full((n,), -1.0, device=dev)
            h = dist.irecv(t2, 1)
            h.wait()
            torch.cuda.synchronize()
            if not torch.equal(t2, torch.full((n,), float(it + 1) * 2, device=dev)):
                bad += 1
        else:
            t = torch.full((n,), -1.0, device=dev)
            h = dist.irecv(t, 0)
            h.wait()
            torch.cuda.synchronize()
            if not torch.equal(t, torch.full((n,), float(it + 1), device=dev)):
                bad += 1
            h = dist.isend(torch.full((n,), float(it + 1) * 2, device=dev), 0)
            h.wait()
print(f"[rank{rank}] errors {bad} of {checks}", flush=True)
dist.destroy_process_group()
