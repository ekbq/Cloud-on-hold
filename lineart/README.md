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

## Mac / GPU

`device.py` picks CUDA, then Apple Silicon (MPS), then CPU, using fp16 on GPU. Override with
`LINEART_DEVICE=cpu|mps|cuda`; `LINEART_STEPS=n` overrides the step count.

`sdxl.py` is the SDXL version (SDXL base + `ShermanG/ControlNet-Standard-Lineart-for-SDXL` + the
fp16-fixed VAE). Same usage as `sdcn3.py`, writes `sdxl_<tag>_*`. Needs a GPU and roughly 10 GB for
the fp16 models, so on a 16 GB Mac close other apps first. **Not yet run anywhere** - it was written
in a CPU-only container that can't fit SDXL.

## Status

`sdcn3.py` = `sdcn2.py` + coloring-book LoRA + subject prompt + Otsu binarize + potrace SVG.
Usage: `python sdcn3.py photo.jpg:people_tag:"short description of the subject"`.

- People: pure black/white SVG, close to marcam in look. Weaker: hair less solid, stray fleece strokes,
  the right girl's wink reads as two closed eyes after binarizing.
- Cats: recognisable but behind marcam (blanket lost to the cutout, blotchy fur).
- The SD1.5 LoRA changed only ~6% of pixels at strength 0.5; the start drawing and ControlNet dominate.
  A stronger base (SDXL + line-art ControlNet) needs a GPU.
- Peak RAM is high on CPU: cutout models are freed before SD loads to avoid OOM kills on 15 GB.
