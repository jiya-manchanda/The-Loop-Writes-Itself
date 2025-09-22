# src/synth_dataset.py
"""
Synthesizes pairs of shapes and intermediate morph masks + a simple textured RGB render.
Save structure:
data/synth/
  pair_000/
    start_mask.png
    end_mask.png
    start_img.png
    end_img.png
    frames/frame_000.png  # intermediate rendered images
    frames/mask_000.png   # corresponding masks
"""

import numpy as np
from skimage.draw import polygon
from PIL import Image, ImageFilter
import os
from pathlib import Path
from .utils import ensure_dir, save_mask, save_rgb
import random
import math

def random_polygon(center, radius, n_vertices=8, irregularity=0.35, spikeyness=0.2):
    """
    Generates a random star-ish polygon around center.
    Returns list of (x, y).
    """
    cx, cy = center
    # generate angle steps
    angle_steps = []
    lower = (2*math.pi / n_vertices) * (1 - irregularity)
    upper = (2*math.pi / n_vertices) * (1 + irregularity)
    sum_ = 0
    for _ in range(n_vertices):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        sum_ += tmp
    # normalize
    k = sum_ / (2*math.pi)
    angle_steps = [s / k for s in angle_steps]

    points = []
    angle = random.uniform(0, 2*math.pi)
    for step in angle_steps:
        r_i = np.clip(random.gauss(radius, spikeyness*radius), radius*0.4, radius*1.6)
        x = cx + r_i * math.cos(angle)
        y = cy + r_i * math.sin(angle)
        points.append((x, y))
        angle += step
    return points

def polygon_to_mask(poly, H, W):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    rr, cc = polygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[rr, cc] = 255
    return mask

def render_textured_rgb(mask, jitter=20):
    """
    Simple texture inside mask: color gradient + noise; returns HxWx3 uint8.
    """
    H, W = mask.shape
    base = np.zeros((H, W, 3), dtype=np.float32)
    # gradient
    gx = np.linspace(0, 1, W)[None, :]
    gy = np.linspace(0, 1, H)[:, None]
    for c in range(3):
        col = np.clip(0.25 + 0.75 * (gx * (c+1)/3 + gy * (3-c)/3), 0, 1)
        base[:, :, c] = col
    # add noise only inside mask
    noise = (np.random.randn(H, W) * 0.2 + 0.5)
    for c in range(3):
        base[:, :, c] *= (0.6 + 0.4 * noise)
    # mask apply
    out = (base * 255).astype(np.uint8)
    # set background to white
    bg = np.ones((H, W, 3), dtype=np.uint8) * 255
    out = np.where(mask[:, :, None] > 0, out, bg)
    # slight blur
    pil = Image.fromarray(out)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.asarray(pil)

def interpolate_point_sets(A, B, t):
    return (1.0 - t) * np.array(A) + t * np.array(B)

def generate_pair(out_dir, H=128, W=128, n_vertices=8, n_frames=8):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # choose centers and radii
    cx, cy = W//2, H//2
    r = min(H, W) * 0.28
    A = random_polygon((cx, cy), r, n_vertices=n_vertices, irregularity=0.4, spikeyness=0.25)
    B = random_polygon((cx, cy), r * random.uniform(0.6, 1.1), n_vertices=n_vertices, irregularity=0.6, spikeyness=0.35)
    start_mask = polygon_to_mask(A, H, W)
    end_mask = polygon_to_mask(B, H, W)
    # save start/end images
    save_mask(start_mask, os.path.join(out_dir, "start_mask.png"))
    save_mask(end_mask, os.path.join(out_dir, "end_mask.png"))
    start_img = render_textured_rgb(start_mask)
    end_img = render_textured_rgb(end_mask)
    save_rgb(start_img, os.path.join(out_dir, "start_img.png"))
    save_rgb(end_img, os.path.join(out_dir, "end_img.png"))

    # make frame folder
    frames_dir = os.path.join(out_dir, "frames")
    Path(frames_dir).mkdir(exist_ok=True)
    # sample consistent point count for interpolation
    n_pts = max(len(A), len(B))
    def resample(poly, n):
        # simple resample by linearizing polygon perimeter
        xs = np.array([p[0] for p in poly] + [poly[0][0]])
        ys = np.array([p[1] for p in poly] + [poly[0][1]])
        d = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        cum = np.concatenate([[0], np.cumsum(d)])
        total = cum[-1]
        samples = np.linspace(0, total, n, endpoint=False)
        newp = []
        for s in samples:
            idx = np.searchsorted(cum, s) - 1
            idx = np.clip(idx, 0, len(xs)-2)
            t = (s - cum[idx]) / (cum[idx+1] - cum[idx] + 1e-9)
            x = (1-t) * xs[idx] + t * xs[idx+1]
            y = (1-t) * ys[idx] + t * ys[idx+1]
            newp.append((x, y))
        return newp

    A_rs = resample(A, n_pts)
    B_rs = resample(B, n_pts)

    # circularly shift B to minimize distance
    def best_shift(a, b):
        bestd = 1e9; besti = 0
        for s in range(len(b)):
            bsh = np.roll(b, s, axis=0)
            d = np.mean(np.linalg.norm(a - bsh, axis=1))
            if d < bestd:
                bestd = d; besti = s
        return besti
    A_arr = np.array(A_rs)
    B_arr = np.array(B_rs)
    shift = best_shift(A_arr, B_arr)
    B_arr = np.roll(B_arr, shift, axis=0)

    for f_idx, t in enumerate(np.linspace(0.0, 1.0, n_frames)):
        pts = interpolate_point_sets(A_arr, B_arr, t)
        mask = polygon_to_mask(pts, H, W)
        rgb = render_textured_rgb(mask)
        save_mask(mask, os.path.join(frames_dir, f"mask_{f_idx:03d}.png"))
        save_rgb(rgb, os.path.join(frames_dir, f"frame_{f_idx:03d}.png"))

def make_dataset(root="data/synth", n_pairs=200, H=128, W=128, n_frames=8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_pairs):
        out = root / f"pair_{i:04d}"
        generate_pair(out, H=H, W=W, n_vertices=random.randint(5,10), n_frames=n_frames)
        if i % 10 == 0:
            print(f"generated {i+1}/{n_pairs}")

if __name__ == "__main__":
    # quick CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    make_dataset(args.out, n_pairs=args.n, H=args.H, W=args.W, n_frames=args.frames)
