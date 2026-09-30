from pathlib import Path

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torchvision import datasets
from concurrent.futures import ThreadPoolExecutor
from torchvision.io import read_image, ImageReadMode


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

class ImageFolderDS(Dataset):
    def __init__(self, base_dir, device='cuda', channels_last=True, workers=16):
        super().__init__()
        # Numeric sort so img_i lines up with row i of noise.pt (lexicographic order would give img_0, img_1, img_10, ...)
        files = sorted(Path(base_dir).glob("*.png"), key=lambda f: int(f.stem.split('_')[-1]))
        read = lambda f: read_image(str(f), mode=ImageReadMode.RGB)
        with ThreadPoolExecutor(workers) as ex:
            imgs = list(ex.map(read, files))

        x = torch.stack(imgs)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        self.x = x.to(device)

        # Noise that generated each image (for rectified flow); noise.pt is padded to a full last batch, so trim it
        noise_path = Path(base_dir) / 'noise.pt'
        if noise_path.exists():
            y = torch.load(noise_path)[:len(files)]
            if channels_last:
                y = y.contiguous(memory_format=torch.channels_last)
            self.y = y.to(device)
        else:
            self.y = torch.zeros(len(files), device=device)
        self.device = device
        self.channels_last = channels_last

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idxs):
        xs = self.x[idxs].float() / 127.5 - 1
        return xs, self.y[idxs]


if __name__ == '__main__':
    a = CifarDS()
    dl = DataLoader(a, sampler=BatchSampler(RandomSampler(a), batch_size=128, drop_last=True), batch_size=None)
    




    