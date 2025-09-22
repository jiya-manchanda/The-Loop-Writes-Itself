import torch
from PIL import Image
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from torchvision.transforms import ToPILImage

def run_controlnet_sequence(mask_images, prompt, out_dir, device='cuda'):
    # controlnet model choice: canny, scribble, or other
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(42)

    prev_image = None
    for i, mask in enumerate(mask_images):
        # mask: numpy uint8 (0/255) -> control image (PIL)
        control_img = Image.fromarray(mask).convert("RGB")
        init_image = None
      
        # Optionally use previous frame as init_image for smoother transitions:
        if prev_image is not None:
            init_image = prev_image

        out = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            image=init_image,
            control_image=control_img,
            strength=0.6,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=generator
        )
        result = out.images[0]
        result.save(f"{out_dir}/frame_{i:03d}.png")
        prev_image = result  # chain for smoother motion
