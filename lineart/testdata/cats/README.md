# Cat test set

- `original.jpg` - source photo (metadata stripped, rotation applied)
- `gemini.jpg` - Gemini line-art version
- `marcam.svg` / `marcam_render.png` - marcam version (the target quality) and a PNG render of it
- `ours/v3.svg` - `v3.py` output (line-art model + threshold + potrace)
- `ours/sd3_s0.5.svg`, `ours/sd3_s0.6.svg` - `sdcn3.py` output at img2img strength 0.5 / 0.6 (CPU, SD1.5)
- `comparison.png` - original / marcam / sd3 0.5 / sd3 0.6 side by side

Run from `lineart/`: `python v3.py testdata/cats/original.jpg cats_v3` then
`python sdcn3.py "testdata/cats/original.jpg:cats:two cats lying on a striped blanket"`.
