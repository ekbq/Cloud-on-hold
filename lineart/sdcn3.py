"""SD1.5 + coloring-book LoRA + ControlNet lineart img2img, then binarize + potrace to SVG.

Usage: python sdcn3.py photo.jpg:tag:"short subject description" [...]
Expects <tag>_v3.png from v3.py as the starting drawing.
"""
import sys, time
import numpy as np, torch, cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from rembg import new_session
import lineart as L, idraw

LORA_REPO = "artificialguybr/coloringbook-redmond-1-5v-coloring-book-lora-for-liberteredmond-sd-1-5"
LORA_FILE = "ColoringBookRedmond15V-LiberteRedmond-ColoringBookAF.safetensors"
LORA_SCALE = 0.8
STRENGTHS = (0.5, 0.6)
PROMPT = ("{subject}, ColoringBookAF, Coloring Book, monochrome lineart, clean black ink outlines on pure "
          "white, bold smooth contour lines, solid black hair, flat, no shading")
NEG = ("photo, realistic, color, grey, gradient, shading, 3d, sketchy, messy, scribbles, noise, blurry, "
       "text, watermark, deformed face, extra fingers")


def finish(img, mask, out):
    """Binarize the generated image, keep ink inside the subject, trace to SVG."""
    g = np.array(img.convert("L"))
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink[cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0] = 0
    ink = L.remove_specks(ink, 25)
    Image.fromarray(255 - ink).save(out + ".png")
    L.to_svg(ink, out + ".svg")
    return 100 * (ink > 0).mean()


if __name__ == "__main__":
    torch.set_num_threads(4)
    cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)
    pipe.fuse_lora(lora_scale=LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    sessions = [new_session("birefnet-general"), new_session("isnet-general-use")]

    for arg in sys.argv[1:]:
        src, tag, subject = arg.split(":", 2)
        im = L.load(src)
        s = 768 / max(im.size)
        w, h = round(im.width * s / 8) * 8, round(im.height * s / 8) * 8
        im = im.resize((w, h), Image.LANCZOS)
        mask = L.subject_mask(im, sessions[0]) | L.subject_mask(im, sessions[1])
        sk = idraw.run("models/rocca_lineart.onnx", im)
        ctrl = ((1 - sk) * 255).astype(np.uint8)
        ctrl[mask == 0] = 0
        control = Image.fromarray(ctrl).convert("RGB")
        init = Image.open(f"{tag}_v3.png").convert("RGB").resize((w, h), Image.LANCZOS)
        for strength in STRENGTHS:
            t = time.time()
            img = pipe(PROMPT.format(subject=subject), negative_prompt=NEG, image=init,
                       control_image=control, strength=strength, num_inference_steps=24,
                       guidance_scale=7.0, controlnet_conditioning_scale=1.0,
                       generator=torch.Generator().manual_seed(7)).images[0]
            base = f"sd3_{tag}_s{strength}"
            img.save(base + "_raw.png")
            pct = finish(img, mask, base)
            print(f"{tag} s{strength}: {w}x{h} {time.time() - t:.0f}s ink {pct:.1f}%", flush=True)
