"""SD1.5 ControlNet-lineart img2img: start from our ink drawing, guided by the masked sketch,
so the generator cleans up line work instead of inventing a photo."""
import sys, time
import numpy as np, torch, cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from rembg import new_session
import lineart as L, idraw

PROMPT = ("monochrome lineart, clean black ink line drawing on pure white paper, coloring book page, "
          "bold smooth outlines, solid black hair and dark markings, flat, no shading")
NEG = ("photo, realistic, photograph, color, grey, gradient, shading, 3d, sketchy, messy, noise, "
       "blurry, text, watermark, sunglasses, deformed face")
torch.set_num_threads(4)
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float32,
    safety_checker=None, requires_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
sessions = [new_session("birefnet-general"), new_session("isnet-general-use")]

for src, tag in [s.split(":") for s in sys.argv[1:]]:
    im = L.load(src)
    s = 768 / max(im.size)
    w, h = round(im.width * s / 8) * 8, round(im.height * s / 8) * 8
    im = im.resize((w, h), Image.LANCZOS)
    mask = sessions[0] and (L.subject_mask(im, sessions[0]) | L.subject_mask(im, sessions[1]))
    sk = idraw.run("models/rocca_lineart.onnx", im)
    ctrl = ((1 - sk) * 255).astype(np.uint8); ctrl[mask == 0] = 0
    control = Image.fromarray(ctrl).convert("RGB")
    init = Image.open(f"{tag}_v3.png").convert("RGB").resize((w, h), Image.LANCZOS)
    for strength in (0.5, 0.75):
        t = time.time()
        img = pipe(PROMPT, negative_prompt=NEG, image=init, control_image=control, strength=strength,
                   num_inference_steps=24, guidance_scale=7.0, controlnet_conditioning_scale=1.0,
                   generator=torch.Generator().manual_seed(7)).images[0]
        img.save(f"sd2_{tag}_s{strength}.png")
        print(f"{tag} strength {strength}: {w}x{h} {time.time()-t:.0f}s", flush=True)
