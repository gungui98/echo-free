import argparse
import os
import wandb
import pytorch_lightning as pl

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from video_diffusion_pytorch import Unet3D, Trainer, KarrasDiffusion

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', type=str, default='echo', choices=['echo', 'dummy'])
    parser.add_argument('--data_dir', type=str, default='camus', choices=['camus', 'echonet', 'udc'])
    parser.add_argument('--chamber_view', type=str, default='2CH', choices=['2CH', '3CH', '4CH'])
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--diffusion_steps', type=int, default=64)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--train_num_steps', type=int, default=100001)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--mask_condition', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print(args)

    wandb = wandb.init(project="video-diffusion", name=f"{args.data_dir}_{args.chamber_view}_{args.num_frames}_{args.image_size}_{'uncond' if args.mask_condition else 'cond'}")
    pl.seed_everything(42)
    results_folder = os.path.join(args.results_folder,f"{args.data_dir}_{args.chamber_view}_{args.num_frames}_{args.image_size}_{'uncond' if args.mask_condition else 'cond'}")
    
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
        data_mode=args.data_mode,  # 'video', 'echo' or 'dummy'
        mask_condition=args.mask_condition,
        train_batch_size=args.batch_size,
        train_lr=1e-3,
        save_and_sample_every=args.log_interval,
        train_num_steps=args.train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        checkpoint=args.checkpoint,  # resume training from checkpoint
        results_folder=results_folder,
    )
    trainer.train()
