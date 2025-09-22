import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
from PIL import Image

def make_interpolated_masks(src_mask, dst_mask, n_frames=20, n_points=200):
    from contour_utils import sample_contour_points
    src_pts = sample_contour_points(src_mask, n_points)
    dst_pts = sample_contour_points(dst_mask, n_points)
  
    # optionally align orientation via circular shift
    # find best circular shift by sampling a subset of k shifts for speed
    def best_shift(a,b,shifts=200):
        best, bi = float('inf'), 0
        for s in range(0, shifts):
            bshift = np.roll(b, s, axis=0)
            d = np.linalg.norm(a - bshift, axis=1).mean()
            if d < best:
                best = d; bi = s
        return bi
    s = best_shift(src_pts, dst_pts, shifts=min(200, n_points))
    dst_pts = np.roll(dst_pts, s, axis=0)

    masks = []
    for t in np.linspace(0.0, 1.0, n_frames):
        interp = (1-t)*src_pts + t*dst_pts
        tform = PiecewiseAffineTransform()
      
        # map src -> interp
        tform.estimate(src_pts, interp)
        warped = warp(src_mask.astype(float)/255.0, tform.inverse, output_shape=src_mask.shape)
        warped_bin = (warped > 0.5).astype('uint8')*255
        masks.append(warped_bin)
    return masks
