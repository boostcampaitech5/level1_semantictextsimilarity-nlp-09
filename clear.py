import torch, gc

## OOM이 발생할 경우, clear해주는 코드
gc.collect()
torch.cuda.empty_cache()