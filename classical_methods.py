import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi
from skimage.filters import threshold_otsu
from skimage.filters import apply_hysteresis_threshold

# image to work on
path = "/home/ebad-atif/Ebad/Sem5/ADIP/project/aircraft-skin-defects-merged-final-1/train/images/hunter_00013_1_jpg.rf.9c822a7f57e8be42d20257ee66709f4e.jpg"

# read image (BGR)
img = cv2.imread(path)

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edge-preserving denoise
denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)

# local contrast boost
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_raw = clahe.apply(gray)        # CLAHE on raw gray (for comparison)
enhanced = clahe.apply(denoised)        # CLAHE after denoise (what we keep)

# edge detection
edges = cv2.Canny(enhanced_raw, 50, 150)

# edge detection on the denoised version, to compare
edges_dn = cv2.Canny(enhanced, 50, 150)

# ridge filter on a float image, targeting thin lines
ridges = frangi(enhanced / 255.0, sigmas=range(1, 4), black_ridges=True)

# threshold the ridge map into a binary mask
thresh = 0.15 * ridges.max()
mask = (ridges > thresh).astype("uint8") * 255

# automatic threshold (Otsu) instead of a hand-picked fraction
t_otsu = threshold_otsu(ridges)
mask_otsu = (ridges > t_otsu).astype("uint8") * 255

# hysteresis: grow the crack from strong seeds into its faint parts
high = t_otsu
low = 0.4 * t_otsu
mask_hyst = apply_hysteresis_threshold(ridges, low, high).astype("uint8") * 255

fig, ax = plt.subplots(1, 9, figsize=(45, 5))
ax[0].imshow(gray, cmap="gray");         ax[0].set_title("grayscale")
ax[1].imshow(enhanced_raw, cmap="gray"); ax[1].set_title("CLAHE (no denoise)")
ax[2].imshow(enhanced, cmap="gray");     ax[2].set_title("denoise + CLAHE")
ax[3].imshow(edges, cmap="gray");        ax[3].set_title("Canny edges")
ax[4].imshow(edges_dn, cmap="gray"); ax[4].set_title("Canny (denoised)")
ax[5].imshow(ridges, cmap="gray", vmin=0, vmax=np.percentile(ridges, 99)); ax[5].set_title("Frangi ridges")
ax[6].imshow(mask, cmap="gray"); ax[6].set_title("crack mask")
ax[7].imshow(mask_otsu, cmap="gray"); ax[7].set_title("crack mask (Otsu)")
ax[8].imshow(mask_hyst, cmap="gray"); ax[8].set_title("crack mask (Hysteresis)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()

