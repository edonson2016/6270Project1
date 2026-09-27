import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torchvision import datasets


class CifarDS(Dataset):
    def __init__(self, train=True, flip=True, device='cuda', channels_last=True):
        super().__init__()
        ds = datasets.CIFAR10(root='./m2/data', train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
        if not channels_last:
            x = x.contiguous()
        self.x = x.to(device)

        self.y = torch.tensor(ds.targets).to(device)
        self.flip = flip
        self.device = device
        self.channels_last = channels_last

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idxs):
        xs = self.x[idxs].float() / 127.5 - 1
        ys = self.y[idxs]

        if self.flip:
            xs = torch.where(torch.rand(len(idxs), device=self.device)[:, None, None, None] < 0.5, xs, xs.flip(3))
        return xs, ys


if __name__ == '__main__':
    a = CifarDS()
    dl = DataLoader(a, sampler=BatchSampler(RandomSampler(a), batch_size=128, drop_last=True), batch_size=None)
    




    