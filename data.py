"""Dataset construction is explicit and lazy."""

def load_datasets(root="/mnt/data", download=False, name="cifar10"):
    from pathlib import Path
    import torchvision
    root = str(Path(root).expanduser())
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10
        normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2470, 0.2435, 0.2616))
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST
        normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    else:
        raise ValueError(f'unsupported dataset {name!r}')
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    def load():
        return (dataset(root=root, train=True, download=download, transform=transform),
                dataset(root=root, train=False, download=download, transform=transform))

    if download:
        # Serialize extraction for co-located macOS/Linux processes.
        import fcntl
        Path(root).mkdir(parents=True, exist_ok=True)
        with (Path(root) / f'.{name}-download.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return load()
    return load()


def load_cifar10(root="/mnt/data", download=False):
    """Convenience entry point for existing CIFAR preparation commands."""
    return load_datasets(root, download, name='cifar10')


def limit_dataset(dataset, count, seed):
    """Select a repeatable subset before sharding, without changing global RNG."""
    import torch
    if count is None or count >= len(dataset):
        return dataset
    if type(count) is not int or count <= 0:
        raise ValueError('dataset sample limit must be positive')
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    return torch.utils.data.Subset(dataset, indices.tolist())


def build_train_loader(dataset, n_workers, worker_rank, batch_size, seed=0, num_workers=2):
    """Equal-length deterministic shards; include padded samples and partial batches.

    DistributedSampler pads to a multiple of worker count. Padded repetitions are
    real processed samples and count in gradient weighting. Keep IDs/seed fixed
    when comparing flat and hierarchical layouts.
    """
    import torch
    if (type(n_workers) is not int or n_workers <= 0 or type(worker_rank) is not int
            or not 0 <= worker_rank < n_workers or type(batch_size) is not int or batch_size <= 0):
        raise ValueError('invalid worker rank/count or batch size')
    if len(dataset) == 0:
        raise ValueError('training dataset must not be empty')
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=n_workers, rank=worker_rank, shuffle=True, seed=seed, drop_last=False)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                      drop_last=False, num_workers=num_workers, pin_memory=False)
