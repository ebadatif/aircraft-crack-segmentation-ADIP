import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi
from skimage.filters import threshold_otsu
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, binary_closing, disk

# image to work on
path = "/home/ebad-atif/Ebad/Sem5/ADIP/project/aircraft-skin-defects-merged-final-1/train/images/hunter_00013_1_jpg.rf.9c822a7f57e8be42d20257ee66709f4e.jpg"

# read image (BGR)
img = cv2.imread(path)

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import os
# draw the ground-truth crack box from the dataset label
label_path = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
H, W = gray.shape
gt = np.dstack([gray, gray, gray]).copy()
if os.path.exists(label_path):
    for line in open(label_path):
        p = line.split()
        if not p or int(float(p[0])) != 0:   # 0 = crack class
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
        x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)
        cv2.rectangle(gt, (x1, y1), (x2, y2), (0, 255, 0), 3)

plt.figure(figsize=(8, 8))
plt.imshow(gt); plt.title("ground-truth crack box"); plt.axis("off")
plt.show()

# # edge-preserving denoise
# denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)

# # local contrast boost
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced_raw = clahe.apply(gray)        # CLAHE on raw gray (for comparison)
# enhanced = clahe.apply(denoised)        # CLAHE after denoise (what we keep)

# # edge detection
# edges = cv2.Canny(enhanced_raw, 50, 150)

# # edge detection on the denoised version, to compare
# edges_dn = cv2.Canny(enhanced, 50, 150)

# # ridge filter on a float image, targeting thin lines
# ridges = frangi(enhanced / 255.0, sigmas=range(1, 4), black_ridges=True)

# # threshold the ridge map into a binary mask
# thresh = 0.15 * ridges.max()
# mask = (ridges > thresh).astype("uint8") * 255

# # automatic threshold (Otsu) instead of a hand-picked fraction
# t_otsu = threshold_otsu(ridges)
# mask_otsu = (ridges > t_otsu).astype("uint8") * 255

# # hysteresis: grow the crack from strong seeds into its faint parts
# high = 0.7 * t_otsu
# low  = 0.25 * t_otsu
# mask_hyst = apply_hysteresis_threshold(ridges, low, high).astype("uint8") * 255

# # clean the mask: bridge small gaps, then drop tiny specks
# m = mask_hyst > 0
# m = binary_closing(m, disk(2))
# m = remove_small_objects(m, min_size=20)
# mask_clean = m.astype("uint8") * 255

# overlay = np.dstack([gray, gray, gray]).copy()
# overlay[mask_clean > 0] = [255, 0, 0]   # red where we detected

# # Figure 1: preprocessing + detection
# stages1 = [
#     ("grayscale", gray),
#     ("CLAHE (no denoise)", enhanced_raw),
#     ("denoise + CLAHE", enhanced),
#     ("Canny edges", edges),
#     ("Canny (denoised)", edges_dn),
#     ("Frangi ridges", ridges),
# ]
# fig1, ax1 = plt.subplots(1, len(stages1), figsize=(28, 5))
# for a, (title, im) in zip(ax1, stages1):
#     if title == "Frangi ridges":
#         a.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(ridges, 99))
#     else:
#         a.imshow(im, cmap="gray")
#     a.set_title(title); a.axis("off")
# fig1.suptitle("Preprocessing & detection")
# plt.tight_layout()

# # Figure 2: thresholding + result
# stages2 = [
#     ("crack mask", mask),
#     ("crack mask (Otsu)", mask_otsu),
#     ("crack mask (Hysteresis)", mask_hyst),
#     ("cleaned mask", mask_clean),
#     ("overlay on image", overlay),
# ]
# fig2, ax2 = plt.subplots(1, len(stages2), figsize=(24, 5))
# for a, (title, im) in zip(ax2, stages2):
#     cmap = None if title.startswith("overlay") else "gray"
#     a.imshow(im, cmap=cmap)
#     a.set_title(title); a.axis("off")
# fig2.suptitle("Thresholding & result")
# plt.tight_layout()

# plt.show()

