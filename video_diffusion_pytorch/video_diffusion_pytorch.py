# helpers functions
import copy
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
import wandb

from video_diffusion_pytorch.dataset import DummyDataset, EchoVideoDataset
from video_diffusion_pytorch.utils import check_shape, unnormalize_img, normalize_img
from video_diffusion_pytorch.utils import is_list_str, exists, default, \
    cycle, noop, num_to_groups
# helpers functions
from video_diffusion_pytorch.utils import video_tensor_to_gif, plot_seq


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.classifer_free_frac = 1.5

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):

        # implement classifer free
        noise = self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        noise_zero = self.denoise_fn.forward_with_cond_scale(x, t, cond=torch.zeros_like(cond), cond_scale=cond_scale)

        noise[:, :3] = noise_zero[:, :3] + self.classifer_free_frac * (noise[:, :3] - noise_zero[:, :3])

        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond,
                                                                 cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond,
                                cond_scale=cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames  # if not exists(cond) else cond.shape[1]
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond=cond,
                                  cond_scale=cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            data_mode="echo",
            mask_condition=False,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100001,
            gradient_accumulate_every=2,
            amp=False,
            checkpoint=None,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results_cond',
            num_sample_rows=1,
            max_grad_norm=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        if data_mode == "echo":
            self.train_ds = EchoVideoDataset(folder, image_size, channels=channels, mask_condition=mask_condition,
                                             num_frames=num_frames, split="train")
            self.test_ds = EchoVideoDataset(folder, image_size, channels=channels, mask_condition=mask_condition,
                                            num_frames=num_frames, split="val")
        elif data_mode == "dummy":
            self.train_ds = DummyDataset(image_size, channels=channels, num_frames=num_frames)
            self.test_ds = DummyDataset(image_size, channels=channels, num_frames=num_frames)
        else:
            raise NotImplementedError()

        print(f'found {len(self.train_ds)} videos for training and {len(self.test_ds)} for testing in {folder}')
        assert len(self.train_ds) > 0 and len(self.test_ds) > 0, 'need to have at least 1 video to start' \
                                                                 ' training (although 1 is not great, try 100k)'

        self.train_dl = cycle(data.DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=False,
                                              pin_memory=True, num_workers=12))
        self.test_dl = cycle(data.DataLoader(self.test_ds, batch_size=train_batch_size, shuffle=False,
                                                pin_memory=True, num_workers=12, drop_last=True))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()
        if checkpoint is not None:
            self.load(checkpoint)
        # check if training using multiple GPUs
        self.multi_gpu = torch.cuda.device_count() > 1
        print(f'using {torch.cuda.device_count()} GPUs for training')
        #
        self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        # self.ema_model = nn.DataParallel(self.ema_model)

    def reset_parameters(self):
        if isinstance(self.model, nn.DataParallel):
            self.ema_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.module.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        wandb.save(str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        # check if state_dict has size mismatch
        filtered_dict = {k: v for k, v in data['model'].items() if k in self.model.state_dict()}
        for k, v in data['model'].items():
            if data['model'][k].shape != self.model.state_dict()[k].shape:
                print(f'Warning: size mismatch for {k} in model, ignoring')
                del filtered_dict[k]
            if "motion_ode" in k:# TODO
                print(f'Warning: {k} in model, ignoring')
                del filtered_dict[k]

        self.model.load_state_dict(filtered_dict, strict=False, **kwargs)

        filtered_dict = {k: v for k, v in data['ema'].items() if k in self.ema_model.state_dict()}
        for k, v in data['ema'].items():
            if data['ema'][k].shape != self.ema_model.state_dict()[k].shape:
                print(f'Warning: size mismatch for {k} in ema model, ignoring')
                del filtered_dict[k]
            
            if "motion_ode" in k:# TODO
                del filtered_dict[k]

        self.ema_model.load_state_dict(filtered_dict, strict=False, **kwargs)
        self.scaler.load_state_dict(data['scaler'])

        print(f'loaded model from step {self.step}')

    def train(
            self,
            prob_focus_present=0.,
            focus_present_mask=None,
            log_fn=noop
    ):
        assert callable(log_fn)
        first_time = True
        progress_bar = tqdm(total=self.train_num_steps, desc='training')
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.train_dl)

                data = {k: v.cuda() for k, v in data.items()}

                with autocast(enabled=self.amp):
                    loss = self.model(
                        data['image'],
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                        cond=data.get('cond', None)[:, 0],
                    )
                    if self.multi_gpu:
                        loss = loss.mean()
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item()})
            log = {'loss': loss.item()}
            wandb.log(log, step=self.step)

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                data = next(self.test_dl)
                while data['image'].shape[0] != self.batch_size:
                    data = next(self.test_dl)
                data = {k: v.cuda() for k, v in data.items()}
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                cond = data.get('cond', None)[:, 0]
                preds  = list(map(lambda n: self.ema_model.sample(batch_size=n, cond=cond), batches))
                
                all_videos_list = torch.cat(preds, dim=0)

                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(all_videos_list, cond, video_path)
                wandb.log({"sample": wandb.Video(video_path, fps=10, format="gif")}, step=self.step)

            self.step += 1

        print('training completed')
