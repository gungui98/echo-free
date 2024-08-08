import copy
import os
from cmath import log
from math import sqrt
from pathlib import Path

import cv2
import einops
import imageio
import matplotlib
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
from functools import partial

from video_diffusion_pytorch.utils import check_shape, normalize_img, unnormalize_img, pseudo_image_from_semantic_map


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t


class KarrasDiffusion(nn.Module):
    """
    An adaptation of the Karras Diffusion model for video denoising.
    """

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
            dynamic_thres_percentile=0.9,

            sigma_min=0.002,  # min noise level
            sigma_max=80,  # max noise level
            sigma_data=0.5,  # standard deviation of data distribution
            rho=7,  # controls the sampling schedule
            P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.classifer_free_frac = 1.5

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile
        # derived preconditioning params - Table 1

    def threshold_x_start(self, x_start, dynamic_threshold=True):
        if not dynamic_threshold:
            return x_start.clamp(-1., 1.)

        s = torch.quantile(
            rearrange(x_start, 'b ... -> b (...)').abs(),
            self.dynamic_thresholding_percentile,
            dim=-1
        )

        s.clamp_(min=1.)
        s = right_pad_dims_to(x_start, s)
        return x_start.clamp(-s, s) / s

    def c_skip(self, sigma_data, sigma):
        return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma ** 2 + sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return torch.log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper
    def preconditioned_network_forward(
            self,
            unet_forward,
            noised_images,
            sigma,
            *,
            sigma_data,
            clamp=False,
            dynamic_threshold=True,
            **kwargs
    ):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1 1').to(device)

        net_out = unet_forward(
            self.c_in(sigma_data, padded_sigma) * noised_images,
            self.c_noise(sigma).to(device),
            **kwargs
        )

        out = self.c_skip(sigma_data, padded_sigma) * noised_images + self.c_out(sigma_data, padded_sigma) * net_out

        if not clamp:
            return out

        return self.threshold_x_start(out, dynamic_threshold)

    def loss_weight(self, sigma_data, sigma):
        return (sigma ** 2 + sigma_data ** 2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, batch_size):
        return (P_mean + P_std * torch.randn((batch_size,))).exp()

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(
            self,
            num_sample_steps,
            rho,
            sigma_min,
            sigma_max
    ):
        N = num_sample_steps
        inv_rho = 1 / rho

        steps = torch.arange(num_sample_steps, dtype=torch.float32)
        sigmas = (sigma_max ** inv_rho + steps / (N - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    def get_sigmas(self):
        return self.sample_schedule(self.num_timesteps, self.rho, self.sigma_min, self.sigma_max)

    @torch.inference_mode()
    def sample(self, pseudo_x=None, cond=None, cond_scale=1.5, batch_size=16, start_step=15):
        device = next(self.denoise_fn.parameters()).device

        batch_size = cond.shape[0] if cond is not None else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames  # if not exists(cond) else cond.shape[1]
        shape = (batch_size, channels, num_frames, image_size, image_size)
        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(self.num_timesteps, self.rho, self.sigma_min, self.sigma_max)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / self.num_timesteps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=device)

        initial_step = 0
        sigmas_and_gammas = sigmas_and_gammas[initial_step:]

        total_steps = len(sigmas_and_gammas)


        pseudo_x = normalize_img(pseudo_x)

        for ind, (sigma, sigma_next, gamma) in tqdm(enumerate(sigmas_and_gammas), total=total_steps,
                                                    desc='sampling time step'):

            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            added_noise = sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            images_hat = images + added_noise



            # create a noise version of cond
            # cond = cond.cpu().numpy()
            # pseudo_cond = matplotlib.cm.viridis(cond[0]/cond.max())
            # cv2.imwrite(f"cond.png", pseudo_cond[:,:,3].astype(np.float32))
            
            if ind == start_step:
                images_hat = pseudo_x + added_noise

            model_output = self.preconditioned_network_forward(
                self.denoise_fn.forward_with_cond_scale,
                images_hat,
                sigma_hat,
                cond=cond, # TODO: change this to cond
                cond_scale=cond_scale,
                sigma_data=self.sigma_data,
            )

            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            has_second_order_correction = sigma_next != 0

            if has_second_order_correction:
                model_output_next = self.preconditioned_network_forward(
                    self.denoise_fn.forward_with_cond_scale,
                    images_next,
                    sigma_next,
                    cond=cond,  # TODO: change this to cond
                    cond_scale=cond_scale, # TODO: change this to cond
                    sigma_data=self.sigma_data,
                )

                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next

            # # visualize the images and pseudo_cond
            # vis_images = torch.cat([images, pseudo_x + added_noise], dim=-2)
            # vis_images = vis_images.clamp(-1., 1.)
            # vis_images = unnormalize_img(vis_images).cpu().numpy()
            # imageio.mimsave(f"./results/{ind}.gif", vis_images[0, 0] * 255, loop=0)

        images = images.clamp(-1., 1.)

        return unnormalize_img(images)

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        x = normalize_img(x)
        # get the sigmas
        sigmas = self.noise_distribution(self.P_mean, self.P_std, b)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1 1').to(device)
        # noise
        noise = torch.randn_like(x)
        noised_images = x + padded_sigmas * noise  # alphas are 1. in the paper
        if hasattr(self.denoise_fn, "reset_loss"):
            self.denoise_fn.reset_loss()

        denoised_images = self.preconditioned_network_forward(
            self.denoise_fn.forward,
            noised_images,
            sigmas,
            sigma_data=self.sigma_data,
            *args,
            **kwargs
        )

        # losses

        losses = F.mse_loss(denoised_images, x, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # loss weighting

        losses = losses * self.loss_weight(self.sigma_data, sigmas).to(device)

        if hasattr(self.denoise_fn, "get_loss"):
            losses += self.denoise_fn.get_loss()

        # return average loss

        return losses.mean()
