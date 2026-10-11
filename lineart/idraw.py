import sys, numpy as np, onnxruntime as o
from PIL import Image, ImageOps
o.set_default_logger_severity(3)
def run(model, im, fixed=None):
    s = o.InferenceSession(model)
    a = np.asarray(im, np.float32) / 255.0
    h, w = a.shape[:2]
    if fixed:
        pad = np.ones((fixed, fixed, 3), np.float32); pad[:h, :w] = a; a = pad
    else:  # dynamic: make dims multiple of 8
        H, W = -(-h // 8) * 8, -(-w // 8) * 8
        pad = np.ones((H, W, 3), np.float32); pad[:h, :w] = a; a = pad
    out = s.run(None, {s.get_inputs()[0].name: a.transpose(2, 0, 1)[None]})[0][0, 0]
    return np.clip(out[:h, :w], 0, 1)
if __name__ == "__main__":
    import lineart as L
    for src, tag in [('../images/4.jpg', 'people'), ('../images/1.jpg', 'cats')]:
        im = L.load(src)
        for m, fx in [('rocca_lineart', None), ('id_anime_1024', 1024), ('id_contour_1024', 1024)]:
            out = run(f'models/{m}.onnx', im, fx)
            Image.fromarray((out * 255).astype(np.uint8)).save(f'raw_{tag}_{m}.png')
            print(tag, m, out.shape, round(float(out.mean()), 3))
