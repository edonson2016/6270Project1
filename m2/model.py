import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) if use_conv else None

    def forward(self, x):
        x = F.interpolate(x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="nearest")
        return self.conv(x) if self.conv is not None else x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, up=False, down=False):
        super().__init__()

        if up:
            block1_conv = Upsample(in_channels, out_channels)
            self.shortcut = Upsample(in_channels, out_channels, use_conv=(in_channels != out_channels))
        elif down:
            block1_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            block1_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if in_channels == out_channels:   
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.block1 = nn.Sequential(
            nn.GroupNorm(16, in_channels), 
            nn.SiLU(), 
            block1_conv,
        )
        self.emb_map = nn.Linear(emb_dim, out_channels)

        block2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        nn.init.zeros_(block2_conv.weight)
        nn.init.zeros_(block2_conv.bias)

        self.block2 = nn.Sequential(
            nn.GroupNorm(16, out_channels), 
            nn.SiLU(), 
            block2_conv
        )

    def forward(self, x, t_emb):
        x_res = self.shortcut(x)
        z = self.block1(x) + self.emb_map(t_emb)[..., None, None]
        return x_res + self.block2(z)


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64, scale=1000.0):
        super().__init__()
        self.dim = dim // 2
        self.scale = scale
        freqs = self.scale ** (torch.arange(self.dim)[None, :] / self.dim)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        return torch.cat((torch.cos(t[:, None] * self.freqs), torch.sin(t[:, None] * self.freqs)), dim=1)


class UNet(nn.Module):
    def __init__(self, init_ch=32, emb_dim=256):
        super().__init__()
        self.t_embedder = TimeEmbedding(dim=emb_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )

        self.conv_beg = nn.Conv2d(3, init_ch, kernel_size=3, stride=1, padding=1)
        
        self.res_down0 = ResBlock(init_ch, init_ch, emb_dim=emb_dim)
        self.res_down1 = ResBlock(init_ch, init_ch*2, emb_dim=emb_dim, down=True)
        self.res_down2 = ResBlock(init_ch*2, init_ch*4, emb_dim=emb_dim, down=True)
        self.res_down3 = ResBlock(init_ch*4, init_ch*8, emb_dim=emb_dim, down=True)

        self.res_mid1 = ResBlock(init_ch*8, init_ch*8, emb_dim=emb_dim)

        self.res_up1 = ResBlock(init_ch*8, init_ch*4, emb_dim=emb_dim, up=True)
        self.res_up2 = ResBlock(init_ch*8, init_ch*2, emb_dim=emb_dim, up=True)
        self.res_up3 = ResBlock(init_ch*4, init_ch, emb_dim=emb_dim, up=True)
        self.res_up4 = ResBlock(init_ch*2, init_ch, emb_dim=emb_dim)
        self.conv_end = nn.Sequential(
            nn.GroupNorm(16, init_ch),
            nn.SiLU(),
            nn.Conv2d(init_ch, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, t):
        x = self.conv_beg(x)
        t_embs = self.t_mlp(self.t_embedder(t))

        x0 = self.res_down0(x, t_embs)
        x1 = self.res_down1(x0, t_embs)
        x2 = self.res_down2(x1, t_embs)
        x3 = self.res_down3(x2, t_embs)

        m = self.res_mid1(x3, t_embs)

        z3 = self.res_up1(m, t_embs)
        z2 = self.res_up2(torch.cat([z3, x2], dim=1), t_embs)
        z1 = self.res_up3(torch.cat([z2, x1], dim=1), t_embs)
        z0 = self.res_up4(torch.cat([z1, x0], dim=1), t_embs)

        return self.conv_end(z0)


        


