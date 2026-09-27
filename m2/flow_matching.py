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

from m2.dataset import CifarDS
from m2.model import UNet

# Fix for cleanfid conflict
import scipy.linalg

_orig_sqrtm = scipy.linalg.sqrtm
def _sqrtm_compat(A, disp=True, **kwargs):
    X = _orig_sqrtm(A, **kwargs)
    return X if disp else (X, None)
scipy.linalg.sqrtm = _sqrtm_compat


def draw_samples(x):
    x0 = torch.randn_like(x)
    t = torch.rand(x0.shape[0], device=x0.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x
    ut = x - x0 
    return xt, t, ut
    
def train_loop(
        epoch,
        config,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        dl,
        device,
        noise
    ):
    n = 0
    loss_sum = torch.zeros((), device=device)
    gn_sum = torch.zeros((), device=device)
    gn_max = torch.zeros((), device=device)
    running_time = time.perf_counter()

    for i, (x, y) in enumerate(dl):
        i_tot = i + epoch * len(dl)
        model.train()
        xt, t, ut = draw_samples(x)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.training.autocast_bf16):
                vt = model(xt, t)

        opt.zero_grad()
        l = F.mse_loss(vt.float(), ut)
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

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
            run_name = f"{config.method}_{config.model.name}_seed{config.training.seed}"
            image_path = Path(config.paths.checkpoint_dir) / run_name / 'images' / f'step_{i_tot}.png'
            image_path.parent.mkdir(parents=True, exist_ok=True)

            samples = eval_samples(model, noise, config.training.log_image_steps)
            imgs = (samples.float().clamp(-1, 1) + 1) / 2                   # to [0, 1]
            grid = torchvision.utils.make_grid(imgs, nrow=10, padding=2, pad_value=1.0)
            grid = (grid * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()   # HWC uint8
            wandb.log({"train/step": i_tot, "samples/grid": wandb.Image(grid, caption=f"step {i_tot}")})

            img = Image.fromarray(grid)
            img.save(image_path)
            model.train()

        if i_tot % config.training.save_interval == 0:
            run_name = f"{config.method}_{config.model.name}_seed{config.training.seed}"
            ckpt_path = Path(config.paths.checkpoint_dir) / run_name / f'ckpt_{i_tot}.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Saving to {ckpt_path}")
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "i_tot": i_tot,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state()
            }
            torch.save(ckpt, ckpt_path)

def eval_samples(model, noise, sample_steps):
    model.eval()
    with torch.no_grad():
        x = noise.clone()
        for step in range(sample_steps):
            x += model(x, torch.full((x.shape[0],), step / sample_steps, device=x.device)) / sample_steps
        return x
    
def train_fm(config_path):
    # Configure wandb
    config = OmegaConf.load(config_path)
    run_name = f"{config.method}_{config.model.name}_seed{config.training.seed}"
    run = wandb.init(
        name=run_name,
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
    if config.training.channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    if config.training.compile:
        model.compile()
    noise = torch.randn(config.training.log_image_num, 3, 32, 32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    for i in range(config.training.epochs):
        train_loop(i, config, model, opt, dl, device, noise)
    wandb.finish()


def eval_fm(train_config_path, eval_config_path):
    train_config = OmegaConf.load(train_config_path)
    eval_config = OmegaConf.load(eval_config_path)
    config = OmegaConf.merge(train_config, eval_config)

    run_name = f"{config.method}_{config.model.name}_seed{config.training.seed}"
    run = wandb.init(
        name=f"eval_{run_name}",
        entity="cis6270",
        project="project1-modality2",
        tags=[config.method, 'eval'],
        config=config,
    )

    # Assign device and seed
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.random.manual_seed(config.eval.seed)

    # Create model
    if config.model.name == 'UNet':
        model = UNet(init_ch=config.model.init_channels, emb_dim=config.model.t_embedding_dim)
    ckpt_path = Path(config.paths.checkpoint_dir) / run_name / f'ckpt_{config.eval.checkpoint_step}.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if config.training.channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    def generator(z):
        batch_size = z.shape[0]
        noise = torch.randn((batch_size, 3, 32, 32), device=device)
        samples = eval_samples(model, noise, config.eval.sample_steps)
        pix_vals = (samples.float().clamp(-1, 1) + 1) / 2
        return (pix_vals * 255).round().to(torch.uint8)

    fid_score = fid.compute_fid(
        gen=generator,
        dataset_name='cifar10',
        dataset_res=32,
        dataset_split='train',
        num_gen=config.eval.num_gen,
        batch_size=config.eval.batch_size,
        device=device
    )
    wandb.log({"fid_score": fid_score})

    wandb.finish()



if __name__ == '__main__':
    train_fm('./m2/configs/base_train_fm.yaml')
    # eval_fm('./m2/configs/base_train_fm.yaml', './m2/configs/base_eval_fm.yaml')



    