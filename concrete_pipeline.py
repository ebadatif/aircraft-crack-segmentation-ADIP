import glob, os
import cv2, numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, closing, disk, dilation
import metrics

img_dir = "/home/ebad-atif/Ebad/Sem5/ADIP/project/concrete-dataset/train_img"
lab_dir = "/home/ebad-atif/Ebad/Sem5/ADIP/project/concrete-dataset/train_lab"

# --- load one image + its mask ---
img_path = sorted(glob.glob(img_dir + "/*"))[0]
stem = os.path.splitext(os.path.basename(img_path))[0]

gt   = cv2.imread(glob.glob(f"{lab_dir}/{stem}.*")[0], cv2.IMREAD_GRAYSCALE)
gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

stages = [("gray", gray)]

# --- pipeline: each stage = compute, then append ---
den = cv2.bilateralFilter(gray, 7, 50, 7)
stages.append(("denoise", den))

enh = cv2.createCLAHE(2.0, (8, 8)).apply(den)
stages.append(("clahe", enh))

ridges = frangi(enh / 255.0, sigmas=range(1, 4), black_ridges=True)
print("frangi max:", round(float(ridges.max()), 5), " nonzero px:", int((ridges > 0).sum()))
stages.append(("frangi", ridges))

t = threshold_otsu(ridges)
pred = apply_hysteresis_threshold(ridges, 0.4 * t, t)
pred = remove_small_objects(closing(pred, disk(2)), 50)
pred = dilation(pred, disk(3))        # thicken thin centreline toward GT band width
pred = pred.astype(np.uint8) * 255
stages.append(("prediction", pred))

# --- score + show (leave this at the bottom) ---
print("SCORES:", {k: round(v, 3) for k, v in metrics.evaluate(pred, gt).items()})

fig, ax = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4))
for a, (title, im) in zip(np.atleast_1d(ax), stages):
    a.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99) or 1)
    a.set_title(title); a.axis("off")
plt.tight_layout()
plt.show()