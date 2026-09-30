import math
from pathlib import Path
import shutil
from omegaconf import OmegaConf

from cleanfid import fid
import torch_fidelity
import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from m2.dataset import CifarDS
from m2.model import UNet
from m2.flow_matching_losses import sample_gmm

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
    def __init__(self, model, steps, gmm=None):
        super().__init__()
        self.model = model
        self.steps = steps
        self.gmm = gmm
        
    def forward(self, z, cast_int=True):
        with torch.no_grad():
            batch_size = z.shape[0]
            if self.gmm is not None:
                noise = sample_gmm(batch_size, **self.gmm).reshape(batch_size, 3, 32, 32)
            else:
                noise = torch.randn((batch_size, 3, 32, 32), device=z.device)
            samples = eval_samples(self.model, noise, self.steps)
            pix_vals = (samples.float().clamp(-1, 1) + 1) / 2
            out = (pix_vals * 255).round().to(torch.uint8) if cast_int else pix_vals
            return out, noise.cpu()
    
    
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

    # Generate and save samples
    gmm = {k: v.to(device) for k, v in ckpt['gmm'].items()} if ckpt.get('gmm') is not None else None
    sampler = Sampler(model, steps=config.eval.sample_steps, gmm=gmm)
    save_dir = Path(config.paths.eval_save_dir) / f"{config.name}_{config.eval.sample_steps}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / f'img_{config.eval.num_gen - 1}.png').exists():
        new_save_dir = save_dir.parent / '_temp'
        shutil.rmtree(new_save_dir, ignore_errors=True)
        new_save_dir.mkdir(parents=True, exist_ok=False)
        for i in range(config.eval.num_gen):
            shutil.copy(save_dir / f"img_{i}.png", new_save_dir / f"img_{i}.png")
        save_dir = new_save_dir

    else:
        image_id = 0
        noise_list = []
        for i in tqdm.tqdm(range(math.ceil(config.eval.num_gen / config.eval.batch_size))):
            samples, noise = sampler(torch.zeros((config.eval.batch_size, 1), device=device), cast_int=False)
            noise_list.append(noise)

            for j in range(samples.size(0)):
                if image_id == config.eval.num_gen:
                    break
                file_path = save_dir / f"img_{image_id}.png"
                save_image(samples[j], file_path)
                image_id += 1

        torch.save(torch.cat(noise_list, dim=0), save_dir / 'noise.pt')

    # fid_score = fid.compute_fid(
    #     gen=sampler.forward,
    #     dataset_name='cifar10',
    #     dataset_res=32,
    #     dataset_split='train',
    #     num_gen=config.eval.num_gen,
    #     batch_size=config.eval.batch_size,
    #     device=device
    # )
    # wandb.log({"fid_score": fid_score})

    if config.eval.num_gen == 50000:
        input2 = "cifar10-train"
    elif config.eval.num_gen == 10000:
        input2 = "cifar10-test"
    metrics = torch_fidelity.calculate_metrics(
        input1=str(save_dir), # torch_fidelity.GenerativeModelModuleWrapper(sampler, 1, 'normal', 0),
        input2=input2,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        verbose=True,
        batch_size=config.eval.batch_size,
        input1_model_num_samples=config.eval.num_gen
    )
    wandb.log(metrics)

    wandb.finish()


if __name__ == '__main__':
    eval_fm('./m2/configs/base_fm.yaml', './m2/configs/ot.yaml')