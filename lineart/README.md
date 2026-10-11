# Photo to line-art experiments

Trying to reproduce the "marcam" style: ink line art that stays faithful to the photo, exported as SVG.

## Pipeline

1. `v3.py` - cutout (rembg: birefnet + isnet, union) -> Informative Drawings line-art model
   (`models/rocca_lineart.onnx`) -> threshold to ink + solid black for the darkest areas + subject
   outline -> potrace SVG. Writes `<out>.png` and `<out>.svg`.
2. `sdcn2.py` - Stable Diffusion 1.5 + ControlNet lineart, img2img. Starts from the `v3.py` ink
   drawing (`<tag>_v3.png`) and is guided by the masked line-art sketch. Strength 0.5 keeps faces and
   expressions; 0.75 starts inventing details. Writes `sd2_<tag>_s<strength>.png`.

`lineart.py` holds shared helpers (load, mask, potrace SVG export) plus an older DexiNed-based
variant. `idraw.py` runs the Informative Drawings ONNX models.

## Run

Run everything from this folder (model paths are relative).

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./get_models.sh
./venv/bin/python v3.py photo.jpg people_v3       # -> people_v3.png / .svg
./venv/bin/python sdcn2.py photo.jpg:people       # -> sd2_people_s0.5.png, sd2_people_s0.75.png
```

On a 4-core CPU, `sdcn2.py` takes about 2.5-4.5 minutes per image at 768px. A GPU is much faster.

## Status

People photos: strength 0.5 is close to marcam and keeps expressions better, but it is greyer and
messier on textured clothing. Cats: still clearly behind marcam. The SD output is not yet
thresholded and traced to SVG.
