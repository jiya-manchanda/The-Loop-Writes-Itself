"""
Minimal conditional denoising model trainer.

- Trains a small UNet (diffusers.UNet2DModel) to predict noise for images conditioned on a mask channel.
- Input tensor: (B, 4, H, W) where channels = RGB + mask
- Uses DDPMScheduler to add noise and MSE loss to predict noise.
- Not a Stable Diffusion fine-tune. Intended for quick experiments on the synth dataset.

Usage example:
python src/train_diffusion.py --data_root data/synth --out_dir models/cond_ddpm --epochs 20 --batch_size 16
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
import numpy as np
from pathlib import Path
from accelerate import Accelerator
import random
from tqdm.auto import tqdm
from .utils import load_mask, load_rgb, ensure_dir

# ----------------------------
# Dataset
# ----------------------------
class CondMorphDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Expects folders pair_**** with start_img.png, end_img.png, frames/mask_***.png, frames/frame_***.png
        We'll sample frames/mask pairs as training examples.
        """
        self.root = Path(root)
        self.pairs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.examples = []
        for p in self.pairs:
            frames_dir = p / "frames"
            if frames_dir.exists():
                masks = sorted(list(frames_dir.glob("mask_*.png")))
                imgs = sorted(list(frames_dir.glob("frame_*.png")))
                for m, im in zip(masks, imgs):
                    self.examples.append((m, im))
        self.transform = transform or (lambda x: x)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        m_path, img_path = self.examples[idx]
        mask = load_mask(m_path)  # HxW
        img = load_rgb(img_path)  # HxWx3
        # normalize
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1,1]
        mask = (mask > 127).astype(np.float32)  # 0/1
        # stack channels: RGB then mask as extra channel
        x = np.concatenate([img.transpose(2, 0, 1), mask[None, :, :]], axis=0).astype(np.float32)
        x = torch.from_numpy(x)
        return x

# ----------------------------
# Training
# ----------------------------
def collate_fn(batch):
    return torch.stack(batch, dim=0)

def train(data_root="data/synth", out_dir="models/cond_ddpm", epochs=20, batch_size=16, lr=1e-4, image_size=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    ensure_dir(out_dir)

    # model: small UNet, conditioning via additional input channels
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=4,         # RGB + mask
        out_channels=3,        # predict RGB noise
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # dataset + loader
    dataset = CondMorphDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        for batch in loop:
            # batch shape: (B, 4, H, W)
            batch = batch.to(accelerator.device)  # float32
            # split into clean image and mask
            clean_rgb = batch[:, :3, :, :]   # [-1,1]
            mask = batch[:, 3:, :, :]        # 0/1

            # sample random timesteps
            bsz = clean_rgb.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=accelerator.device).long()
            noise = torch.randn_like(clean_rgb)

            # add noise according to scheduler
            noisy = scheduler.add_noise(clean_rgb, noise, timesteps)

            # prepare model input: concatenate noisy RGB + mask channel
            model_input = torch.cat([noisy, mask], dim=1)

            noise_pred = model(model_input, timesteps).sample  # predicted noise (3 channels)

            loss = mse(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/ (loop.n if loop.n else 1))

        # save checkpoint at epoch end
        if accelerator.is_main_process:
            ckpt = {"model_state_dict": accelerator.get_state_dict(model)}
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_epoch_{epoch+1}.pt"))
            print(f"Saved checkpoint to {out_dir}/ckpt_epoch_{epoch+1}.pt")

    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/synth")
    parser.add_argument("--out_dir", default="models/cond_ddpm")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()
    train(args.data_root, args.out_dir, args.epochs, args.batch_size, args.lr, args.image_size)
