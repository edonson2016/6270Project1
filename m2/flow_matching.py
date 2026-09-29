from pathlib import Path
import time
from omegaconf import OmegaConf

from PIL import Image
from cleanfid import fid
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
import torchvision
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from sklearn.mixture import GaussianMixture # type: ignore

from m2 import flow_matching_losses
from m2.dataset import CifarDS
from m2.model import UNet
from m2.flow_matching_eval import eval_samples


def pretrain_gmm(dl, config):
    xs, n = [], 0
    for x, _ in dl:
        xs.append(x.flatten(1).float().cpu())
        n += x.shape[0]
        if n >= config.gmm.num_fit_samples:
            break
    X = torch.cat(xs)[:config.gmm.num_fit_samples].numpy()

    gmm = GaussianMixture(
        n_components=config.gmm.components,
        covariance_type=config.gmm.covariance_type,
        random_state=config.training.seed,
    )
    gmm.fit(X)
    return {
        'means': torch.from_numpy(gmm.means_).float(),
        'covs': torch.from_numpy(gmm.covariances_).float(),
        'weights': torch.from_numpy(gmm.weights_).float(),
    }


def train_loop(
        epoch,
        config,
        model: torch.nn.Module,
        ema,
        opt: torch.optim.Optimizer,
        dl,
        device,
        noise,
        i_tot_start=0,
        **kwargs,
    ):
    n = 0
    loss_sum = torch.zeros((), device=device)
    gn_sum = torch.zeros((), device=device)
    gn_max = torch.zeros((), device=device)
    running_time = time.perf_counter()

    for i, (x, y) in enumerate(dl):
        i_tot = i + epoch * len(dl) + i_tot_start
        model.train()

        opt.zero_grad()
        l = flow_matching_losses.loss_method_dict[config.loss.type](x, model, device, config, **kwargs)
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_gradient_norm)
        opt.step()
        if ema is not None:
            ema.update_parameters(model)
            eval_model = ema
        else:
            eval_model = model

        n += 1
        loss_sum += l.detach()
        gn_sum += gn.detach()
        gn_max = torch.maximum(gn_max, gn.detach())
        
        if i % config.training.log_interval == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            wandb.log({
                'train/epoch': epoch,
                "train/step": i_tot, 
                "train/loss": (loss_sum / n).item(), 
                "train/grad_norm": (gn_sum / n).item(), 
                "train/grad_max": gn_max.item(),
                "perf/time_per_step": (time.perf_counter() - running_time) / n,
                'perf/cuda_space': torch.cuda.memory_allocated() / 1e9
            })
            n = 0; running_time = time.perf_counter()
            loss_sum.zero_(); gn_sum.zero_(); gn_max.zero_()

        if i_tot % config.training.log_image_interval == 0:
            image_path = Path(config.paths.checkpoint_dir) / config.name / 'images' / f'step_{i_tot}.png'
            image_path.parent.mkdir(parents=True, exist_ok=True)

            samples = eval_samples(eval_model, noise, config.training.log_image_steps)
            imgs = (samples.float().clamp(-1, 1) + 1) / 2                   # to [0, 1]
            grid = torchvision.utils.make_grid(imgs, nrow=10, padding=2, pad_value=1.0)
            grid = (grid * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()   # HWC uint8
            wandb.log({"train/step": i_tot, "samples/grid": wandb.Image(grid, caption=f"step {i_tot}")})

            img = Image.fromarray(grid)
            img.save(image_path)

        if i_tot % config.training.save_interval == 0:
            ckpt_path = Path(config.paths.checkpoint_dir) / config.name / f'ckpt_{i_tot}.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            ema_state_dict = ema.module.state_dict() if ema is not None else None
            print(f"Saving to {ckpt_path}")
            ckpt = {
                "model": model.state_dict(),
                "ema": ema_state_dict,
                "opt": opt.state_dict(),
                "i_tot": i_tot,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "gmm": {k: v.cpu() for k, v in kwargs.items()} if kwargs else None,
            }
            torch.save(ckpt, ckpt_path)

    
def train_fm(config_path, added_config_path=None):
    # Configure wandb
    config = OmegaConf.load(config_path)
    if added_config_path is not None:
        added_config = OmegaConf.load(added_config_path)
        config = OmegaConf.merge(config, added_config) # latter overrides

    run = wandb.init(
        name=config.name,
        entity="cis6270",
        project="project1-modality2",
        tags=[config.method, 'train'],
        config=config,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")

    # Assign device and seed
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(config.training.seed)

    # Create data, model, opt
    a = CifarDS(train=True, flip=config.data.flip, device=device, channels_last=config.training.channels_last)
    dl = DataLoader(a, sampler=BatchSampler(RandomSampler(a), batch_size=config.training.batch_size, drop_last=True), batch_size=None)

    if config.model.name == 'UNet':
        model = UNet(init_ch=config.model.init_channels, emb_dim=config.model.t_embedding_dim)

    if config.training.get("checkpoint_path") is not None:
        ckpt = torch.load(config.training.checkpoint_path, map_location='cpu')
    else:
        ckpt = None

    if ckpt is not None:
        model.load_state_dict(ckpt['model'])

    if config.training.channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    if config.training.use_ema:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.training.ema_decay), use_buffers=True)
        ema.requires_grad_(False)
        if ckpt is not None:
            ema.module.load_state_dict(ckpt['ema'])
            ema.n_averaged.fill_(1)
    else:
        ema = None

    if config.training.compile:
        model.compile()

    gmm = {}
    if config.loss.type == 'gmm':
        gmm = ckpt['gmm'] if ckpt is not None else pretrain_gmm(dl, config)
        gmm = {k: v.to(device) for k, v in gmm.items()}
        noise = flow_matching_losses.sample_gmm(config.training.log_image_num, **gmm).reshape(-1, 3, 32, 32)
    else:
        noise = torch.randn(config.training.log_image_num, 3, 32, 32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    if ckpt is not None:
        opt.load_state_dict(ckpt['opt'])
        i_tot_start = ckpt['i_tot']
    else:
        i_tot_start = 0
    
    for i in range(config.training.epochs):
        train_loop(i, config, model, ema, opt, dl, device, noise, i_tot_start=i_tot_start, **gmm)
    wandb.finish()




if __name__ == '__main__':
    train_fm('./m2/configs/base_fm.yaml')




    