"""Photo -> line art via Informative Drawings (rocca ONNX) -> ink -> potrace SVG."""
import sys, cv2, numpy as np
from PIL import Image
from rembg import new_session
import lineart as L, idraw
LINE_THR, DARK_PCT, MIN_STROKE = 0.78, 6, 70
src, out = sys.argv[1], sys.argv[2]
im = L.load(src); rgb = np.array(im)
# union of two cutout models: each misses different parts of the subjects
mask = L.subject_mask(im, new_session("birefnet-general")) | L.subject_mask(im, new_session("isnet-general-use"))
sk = idraw.run('models/rocca_lineart.onnx', im)
lines = ((sk < LINE_THR) & (mask > 0)).astype(np.uint8) * 255
lines = L.remove_specks(lines, MIN_STROKE)  # drops short fleece/fur texture strokes
g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
dark = ((g < np.percentile(g[mask > 0], DARK_PCT)) & (mask > 0)).astype(np.uint8) * 255
dark = L.remove_specks(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), 80)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
outline = np.zeros_like(mask); cv2.drawContours(outline, cnts, -1, 255, 3)
outline[:3] = outline[-3:] = 0; outline[:, :3] = outline[:, -3:] = 0
ink = lines | dark | outline
Image.fromarray(255 - ink).save(out + ".png")
L.to_svg(ink, out + ".svg")
print(out, f"ink {100*(ink>0).mean():.1f}%")
