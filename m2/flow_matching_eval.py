from pathlib import Path
from omegaconf import OmegaConf

from cleanfid import fid
import torch_fidelity
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2.dataset import CifarDS
from m2.model import UNet

# Fix for cleanfid conflict
import scipy.linalg

_orig_sqrtm = scipy.linalg.sqrtm
def _sqrtm_compat(A, disp=True, **kwargs):
    X = _orig_sqrtm(A, **kwargs)
    return X if disp else (X, None)
scipy.linalg.sqrtm = _sqrtm_compat



def eval_samples(model, noise, sample_steps):
    model.eval()
    with torch.no_grad():
        x = noise.clone()
        for step in range(sample_steps):
            x += model(x, torch.full((x.shape[0],), step / sample_steps, device=x.device)) / sample_steps
        return x


class Sampler(nn.Module):
    def __init__(self, model, steps):
        super().__init__()
        self.model = model
        self.steps = steps
        
    def forward(self, z):
        batch_size = z.shape[0]
        noise = torch.randn((batch_size, 3, 32, 32), device=z.device)
        samples = eval_samples(self.model, noise, self.steps)
        pix_vals = (samples.float().clamp(-1, 1) + 1) / 2
        return (pix_vals * 255).round().to(torch.uint8)
    
    
def eval_fm(config_path, added_config_path=None):
    config = OmegaConf.load(config_path)
    if added_config_path is not None:
        added_config = OmegaConf.load(added_config_path)
        config = OmegaConf.merge(config, added_config) # latter overrides

    run = wandb.init(
        name=f"eval_{config.name}",
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
    ckpt_path = Path(config.paths.checkpoint_dir) / config.name / f'ckpt_{config.eval.checkpoint_step}.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['ema'] if ckpt.get('ema') is not None else ckpt['model'])
    if config.training.channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    sampler = Sampler(model, steps=config.eval.sample_steps)
    fid_score = fid.compute_fid(
        gen=sampler.forward,
        dataset_name='cifar10',
        dataset_res=32,
        dataset_split='train',
        num_gen=config.eval.num_gen,
        batch_size=config.eval.batch_size,
        device=device
    )
    sampler = Sampler(model, steps=config.eval.sample_steps)
    metrics = torch_fidelity.calculate_metrics(
        input1=torch_fidelity.GenerativeModelModuleWrapper(sampler, 1, 'normal', 0),
        input2="cifar10-train",
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        verbose=True,
        batch_size=config.eval.batch_size,
        input1_model_num_samples=config.eval.num_gen
    )


    wandb.finish()


if __name__ == '__main__':
    eval_fm('./m2/configs/base_fm.yaml')