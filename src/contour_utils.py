import cv2, numpy as np

def sample_contour_points(mask, n_points=200):
    # mask: binary numpy array
    mask_u8 = (mask>0).astype('uint8')*255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
      
    # choose the largest contour
    cnt = max(contours, key=lambda c: cv2.contourArea(c))
    cnt = cnt.squeeze()
  
    # compute cumulative arc-length
    diffs = np.sqrt(((np.diff(cnt, axis=0))**2).sum(axis=1))
    dists = np.concatenate([[0], np.cumsum(diffs)])
    total = dists[-1]
    if total == 0:
        return cnt[:n_points]
    samples = np.linspace(0, total, n_points, endpoint=False)
  
    # place samples by linear interpolation along contour
    res = []
    j = 0
    for s in samples:
        while j < len(dists)-1 and dists[j+1] < s:
            j += 1
        if j == len(dists)-1:
            res.append(cnt[-1])
        else:
            t = (s - dists[j]) / (dists[j+1]-dists[j]+1e-8)
            p = (1-t)*cnt[j] + t*cnt[j+1]
            res.append(p)
    return np.array(res)  # shape (n_points, 2)
