# Aircraft Skin Crack Segmentation

Finding cracks on aircraft skin using classical image processing — no deep learning in this part.
This is the front-end of a larger plan: a YOLO detector will later propose regions, and this
classical pipeline does the pixel-level crack segmentation inside them. Right now the classical
pipeline is what's built.

It's a university project (Applied Digital Image Processing) that we're also pushing toward a
short paper.

## The problem

Aircraft skin photos are hard. The metal is a big, uniform, dull-grey surface; the lighting is
uneven (glare on curved panels, shadow near rivets); and a crack is just a thin, faint, dark line
that's only slightly darker than the skin around it. On top of that the images are full of things
that *look* like cracks but aren't — rivet rows, panel seams, and water/dirt streaks. So the real
challenge isn't only finding the crack, it's not getting fooled by everything else.

## The pipeline so far

We built it one stage at a time, checking the result at each step rather than trusting defaults:

1. **Grayscale** — a crack is about brightness, not colour.
2. **Bilateral denoise** — smooths the noisy surface texture while keeping crack edges sharp
   (a normal blur would wash the crack out). We confirmed this was worth it by comparing the
   edge maps with and without it.
3. **CLAHE** — boosts local contrast so faint cracks separate from the skin, without blowing out
   the bright areas.
4. **Frangi ridge filter** — instead of finding every edge, this responds to thin *line-like*
   structures, which is what a crack actually is. It ignores big step-edges like hatches and pipes.
5. **Hysteresis threshold** — turns the ridge response into a yes/no mask. A single cutoff didn't
   work (the crack is strong in one spot, faint elsewhere), so we grow the crack out from strong
   "seed" pixels into its own faint continuation.

Each run shows all the stages side by side, so you can see the image get cleaner step by step.

## What's next

- Morphological cleanup — drop tiny specks, bridge small gaps in the crack.
- Seam / streak suppression — the main contribution: use geometry (straight-line detection) to
  remove rivet rows, seams and streaks, which are the hardest false positives.
- Evaluation against hand-traced masks (Dice / IoU) on the test split.

## Files

- `stage1_data.py` — downloads the dataset from Roboflow (use your own API key)
- `classical_methods.py` — the pipeline above, stage by stage

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The dataset isn't committed (see `.gitignore`) — pull it yourself with `stage1_data.py`.
Source: Roboflow `aircraft-skin-defects-merged-final` (CC BY 4.0), 5 classes —
crack, dent, missing-head, paint-off, scratch. Only the crack class matters here.

## Citation

Dataset, via Roboflow Universe (CC BY 4.0):

```bibtex
@misc{ aircraft-skin-defects-merged-final_dataset,
  title = { aircraft-skin-defects-merged-final Dataset },
  type = { Open Source Dataset },
  author = { Dibya Dillip },
  howpublished = { \url{ https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-merged-final } },
  url = { https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-merged-final },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2023 },
  month = { jun },
  note = { visited on 2026-10-20 },
}
```