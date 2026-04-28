import torch

def infinite_iterator(loader, distributed=False, sampler=None):
    while True:
        if distributed and sampler is not None:
            sampler.set_epoch(torch.randint(0, 100000, (1,)).item())
        for batch in loader:
            yield batch