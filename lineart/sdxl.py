"""SDXL + lineart ControlNet img2img, then binarize + potrace to SVG. Needs a GPU (CUDA or Apple MPS).

Usage: python sdxl.py photo.jpg:tag:"short subject description" [...]
Expects <tag>_v3.png from v3.py as the starting drawing. Writes sdxl_<tag>_s<strength>.{png,svg}.

Memory: roughly 10 GB for the fp16 models; on a 16 GB Mac close other apps first.
"""
import os, sys, time
import cv2, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
import device as D
from sdcn3 import NEG, PROMPT, finish, prepare_jobs

STRENGTHS = (0.5, 0.6)
STEPS = int(os.environ.get("LINEART_STEPS", 30))
LONG_SIDE = 1024
PROMPT_XL = PROMPT.replace("ColoringBookAF, Coloring Book, ", "coloring book page, ")


def lineart_standard(rgb, sigma=6.0, threshold=8):
    """controlnet_aux's LineartStandardDetector, the preprocessor this ControlNet was trained on.

    White lines on black: how much darker each pixel is than its blurred surroundings.
    """
    x = rgb.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), sigma)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > threshold]))
    intensity *= 127
    return intensity.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    jobs = prepare_jobs(sys.argv[1:], LONG_SIDE)

    dev, dtype = D.pick()
    print(f"device: {dev} ({dtype})", flush=True)
    if dev == "cpu" and not os.environ.get("LINEART_ALLOW_CPU"):
        sys.exit("SDXL on CPU needs ~20 GB RAM and many minutes per image; "
                 "set LINEART_ALLOW_CPU=1 to try anyway, or use sdcn3.py.")
    cn = ControlNetModel.from_pretrained("ShermanG/ControlNet-Standard-Lineart-for-SDXL", torch_dtype=dtype)
    # the stock SDXL VAE produces black/NaN images in fp16; this one is patched for it
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae, torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None, use_safetensors=True)
    pipe.set_progress_bar_config(disable=True)
    pipe = D.prepare(pipe, dev)

    for tag, subject, im, w, h, mask in jobs:
        ctrl = lineart_standard(np.array(im))
        ctrl[mask == 0] = 0
        control = Image.fromarray(ctrl).convert("RGB")
        control.save(f"sdxl_{tag}_control.png")
        init = Image.open(f"{tag}_v3.png").convert("RGB").resize((w, h), Image.LANCZOS)
        for strength in STRENGTHS:
            t = time.time()
            img = pipe(PROMPT_XL.format(subject=subject), negative_prompt=NEG, image=init,
                       control_image=control, strength=strength, num_inference_steps=STEPS,
                       guidance_scale=7.0, controlnet_conditioning_scale=0.9,
                       width=w, height=h, generator=torch.Generator().manual_seed(7)).images[0]
            base = f"sdxl_{tag}_s{strength}"
            img.save(base + "_raw.png")
            pct = finish(img, mask, base)
            print(f"{tag} s{strength}: {w}x{h} {time.time() - t:.0f}s ink {pct:.1f}%", flush=True)
