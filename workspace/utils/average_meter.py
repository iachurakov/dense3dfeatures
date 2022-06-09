import torch


class AverageMeter:
    def __init__(self, name, rank, world_size):
        self.name = name
        self.sum = 0
        self.rank = rank
        self.world_size = world_size
        self.count = torch.LongTensor([0]).to(rank)
        self.out = [torch.zeros(2).to(rank) for _ in range(self.world_size)]

    def update(self, val, n=1):
        if isinstance(val, float):
            val = torch.FloatTensor([val]).to(self.rank)
        self.sum += val.detach()
        self.count += n

    def gather(self):
        torch.distributed.all_gather(self.out, torch.hstack([self.sum, self.count]))

    @property
    def avg(self):
        stats = torch.stack(self.out, dim=0).sum(dim=0)
        return (stats[0] / stats[1]).item()

    def __str__(self):
        return f'{self.name}={self.avg:.5f}'


class MultipleAverageMeters:
    def __init__(self, rank, world_size):
        self.meters = {}
        self.rank = rank
        self.world_size = world_size

    def update(self, new_val, n=1):
        for key, val in new_val.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(key, self.rank, self.world_size)
            self.meters[key].update(val, n)

    def gather(self):
        for meter in self.meters.values():
            meter.gather()

    def __str__(self):
        return ', '.join([str(m) for m in self.meters.values()])

    def __getitem__(self, item):
        return self.meters[item].avg

    @property
    def items(self):
        return self.meters.items()