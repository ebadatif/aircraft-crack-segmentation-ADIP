"""Run the tuned pipeline over the whole DeepCrack TEST set and average the metrics.
This is the reportable number (tuning was done on a train image; we report on test).
"""

import glob, os
import cv2, numpy as np
from skimage.filters import frangi, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, closing, dilation, disk
import metrics

BASE = "/home/ebad-atif/Ebad/Sem5/ADIP/project/concrete-dataset"   # <-- set to the SAME path your concrete.py uses (e.g. "concrete" or "data/concrete")


USE_CLAHE = False      # top of file

# def segment(gray):
#     den = cv2.bilateralFilter(gray, 7, 50, 7)
#     base = cv2.createCLAHE(2.0, (8, 8)).apply(den) if USE_CLAHE else den
#     ridges = frangi(base / 255.0, sigmas=range(1, 4), black_ridges=True)
#     t = threshold_otsu(ridges)
#     m = apply_hysteresis_threshold(ridges, 0.3 * t, 0.7 * t)
#     m = remove_small_objects(closing(m, disk(2)), 50)
#     m = dilation(m, disk(1))
#     return m.astype(np.uint8) * 255

def segment_canny(gray):
    """Method (a): Canny edges gating adaptive threshold."""
    den = cv2.bilateralFilter(gray, 7, 50, 7)
    edges = cv2.Canny(den, 50, 150)
    adap = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 8)
    m = (adap > 0) & dilation(edges > 0, disk(3))
    m = remove_small_objects(closing(m, disk(2)), 200)
    m = dilation(m, disk(1))
    return m.astype(np.uint8) * 255


img_paths = sorted(glob.glob(f"{BASE}/test_img/*"))
print("found", len(img_paths), "test images")

results = []
for i, ip in enumerate(img_paths):
    stem = os.path.splitext(os.path.basename(ip))[0]
    lab = glob.glob(f"{BASE}/test_lab/{stem}.*")
    if not lab:
        continue
    gray = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY)
    gt = cv2.imread(lab[0], cv2.IMREAD_GRAYSCALE)
    if gt.shape != gray.shape:
        gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    results.append(metrics.evaluate(segment_canny(gray), gt))
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(img_paths)} done")

print("\nscored", len(results), "images")
print("MEAN:", {k: round(v, 3) for k, v in metrics.evaluate_many(results).items()})