#!/usr/bin/env python3
"""跨机 collective smoke：2 进程 all_reduce 数值校验（nccl/gloo 可切）。

laptop:  NCCL_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1 \
         python scripts/nccl_smoke_2node.py 0 100.96.154.1 29601 [nccl|gloo]
white:   NCCL_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1 \
         python scripts/nccl_smoke_2node.py 1 100.96.154.1 29601 [nccl|gloo]
"""

import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(sys.argv[1])
    master = sys.argv[2]
    port = int(sys.argv[3])
    backend = sys.argv[4] if len(sys.argv) > 4 else "nccl"
    world = 2
    dev = "cpu"
    if backend == "nccl":
        print(f"[rank {rank}] torch={torch.__version__} cuda={torch.version.cuda} "
              f"nccl={torch.cuda.nccl.version()} dev={torch.cuda.get_device_name(0)}", flush=True)
        torch.cuda.set_device(0)
        dev = "cuda"
    print(f"[rank {rank}] init_process_group({backend}) -> tcp://{master}:{port}", flush=True)
    dist.init_process_group(
        backend, rank=rank, world_size=world, init_method=f"tcp://{master}:{port}"
    )
    print(f"[rank {rank}] pg ready, alloc tensor on {dev} ...", flush=True)
    t = torch.full((4096,), float(rank + 1), device=dev)
    print(f"[rank {rank}] tensor ready, all_reduce ...", flush=True)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(float(r + 1) for r in range(world))  # 1 + 2 = 3
    ok = bool(torch.allclose(t, torch.full_like(t, expected)))
    print(
        f"[smoke rank {rank}] backend={backend} sum={t[0].item()} "
        f"expected={expected} ok={ok}",
        flush=True,
    )
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
