import argparse
import os
import cv2
import einops
import matplotlib
import pytorch_lightning as pl

from video_diffusion_pytorch.utils import match_histograms, pseudo_image_from_semantic_map

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import glob
import torch
import imageio
from einops import rearrange
from torch.nn.functional import interpolate
from tqdm import tqdm

from video_diffusion_pytorch import Unet3D, Trainer, KarrasDiffusion


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', type=str, default='echo', choices=['echo', 'dummy'])
    parser.add_argument('--data_dir', type=str, default='camus', choices=['camus', 'echonet', 'udc'])
    parser.add_argument('--chamber_view', type=str, default='2CH', choices=['2CH', '3CH', '4CH'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--diffusion_steps', type=int, default=64)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--train_num_steps', type=int, default=100001)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--is_sdedit', type=bool, default=False, help="Whether to use original SDEdit or using optimal transport")
    return parser.parse_args()


def main():
    args = get_args()
    pl.seed_everything(5)
    results_folder = os.path.join(args.results_folder,
                                  f"{args.data_dir}_{args.chamber_view}_{args.num_frames}_{args.image_size}")

    model = Unet3D(
        dim=32,
        dim_mults=(1, 2, 4),
    )

    diffusion = KarrasDiffusion(
        model,
        image_size=args.image_size,
        num_frames=args.num_frames,
        timesteps=args.diffusion_steps,  # number of steps
        loss_type='l1'  # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        folder=os.path.join('data', args.data_dir, args.chamber_view),
        data_mode="dummy",  # 'video', 'echo' or 'dummy'
        train_batch_size=args.batch_size,
        train_lr=1e-4,
        save_and_sample_every=args.log_interval,
        train_num_steps=args.train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        checkpoint=None,  # this is for resuming training, for now we don't need it
        results_folder=results_folder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.data_dir = "echonet"
    weight_path = f"weights/{args.data_dir}_model.pt"
    reference_image = cv2.imread(f"images/{args.data_dir}.png", 0)
    state_dict = torch.load(weight_path, map_location=device)
    trainer.ema_model.load_state_dict(state_dict["ema"])
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    for start_step in range(0, 30, 2):
        for idx, seg_path in tqdm(enumerate(glob.glob(os.path.join(f"./visualize/{args.data_dir}", "*_segmap.png")))):
            print(seg_path)
            cond = imageio.imread(seg_path)
            if args.is_sdedit:
                pseudo_cond = matplotlib.cm.get_cmap('tab20')(cond)[..., :3]
                pseudo_cond = cv2.resize(pseudo_cond, (args.image_size, args.image_size)).astype('float32') * 255
            else:
                pseudo_cond = pseudo_image_from_semantic_map(cond, (args.image_size, args.image_size), is_camus = args.data_dir == "camus")
                pseudo_cond = match_histograms(pseudo_cond, reference_image)
            cond = torch.from_numpy(cond) 
            pseudo_cond = torch.from_numpy(pseudo_cond) / 255

            # resizes the condition to the same size as the image
            cond_small = interpolate(cond[None, None, ...].float(), size=(args.image_size, args.image_size),
                                    mode="nearest").squeeze()
            cond_small = torch.repeat_interleave(cond_small[None, ...], args.batch_size, dim=0).long()
            if args.is_sdedit:
                pseudo_cond = einops.repeat(pseudo_cond, 'h w c -> b c t h w',b=args.batch_size, t=args.num_frames)
            else:
                pseudo_cond = einops.repeat(pseudo_cond, 'h w -> b c t h w',b=args.batch_size, c=3, t=args.num_frames)
            pseudo_cond = pseudo_cond.to(device)
            if args.data_dir == "camus":
                cond_small[cond_small >= 1] = 1
            else:
                cond_small = torch.zeros_like(cond_small, dtype=torch.long) 
            output = trainer.ema_model.sample(pseudo_x=pseudo_cond, batch_size=args.batch_size, cond=cond_small.to(device), start_step=start_step)

            output = rearrange(output, 'b c t h w -> b t h w c')
            images = output.detach().cpu().numpy()
            images = (images * 255).astype('uint8')

            patient_id = os.path.basename(seg_path).split("_")[0]
            write_path = f"./visualize/{args.data_dir}/{patient_id}_{start_step}_free" + ("_rgb" if is_rgb else "")
            os.makedirs(write_path, exist_ok=True)
            for i, img in enumerate(images[0]):
                imageio.imwrite(os.path.join(write_path, f"{i:04d}.png"), img)
            imageio.mimwrite(os.path.join(args.results_folder, f"{idx}.gif"), images[0], loop=0)
            print(f"Saved gif file to {os.path.join(args.results_folder, f'{idx}.gif')}")


if __name__ == '__main__':
    main()
