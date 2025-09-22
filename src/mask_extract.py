import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def extract_mask(image_path, out_path, blur=5, thresh_blocksize=51, c=10):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to be robust to hand-drawn shading
    blur_img = cv2.GaussianBlur(gray, (blur, blur), 0)
    mask = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, thresh_blocksize, c)
    # morphological clean
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
  
    # keeping largest connected component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = int(np.argmax(areas))
        mask2 = np.zeros_like(mask)
        cv2.drawContours(mask2, contours, max_idx, 255, -1)
        mask = mask2
    Image.fromarray(mask).save(str(out_path))
    return mask

if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    out = Path(sys.argv[2])
    extract_mask(p, out)
