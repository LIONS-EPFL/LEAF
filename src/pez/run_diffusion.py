import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel




if __name__ == '__main__':


    image_length = 512
    num_images = 4
    guidance_scale = 9
    num_inference_steps = 25
    robust = False
    device = "cuda:0"

    # set seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not robust:
        reconstructions_file = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                            "results-coco-img-100smpls-3000iters-ViT-L-14-quickgelu-clean.json")
        img_out_dir = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                       "images-coco-clean/")
    else:
        reconstructions_file = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                            "results-coco-img-100smpls-3000iters-ViT-L-14-quickgelu-robust.json")
        img_out_dir = ("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-ViT-L-14-quickgelu/"
                       "images-coco-robust/")
    os.makedirs(img_out_dir, exist_ok=True)

    # diffusion_model_id = "stabilityai/stable-diffusion-2-1-base"
    diffusion_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(diffusion_model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        diffusion_model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        # revision="fp16",
        )
    pipe = pipe.to(device)

    if robust:
        # replace the text encoder with our robust one
        # model = CLIPTextModel.from_pretrained("RCLIP/OpenCLIP-ViT-H-rho50-k1-constrained")
        model = CLIPTextModel.from_pretrained("RCLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2")
        pipe.text_encoder = model.to(device,dtype=pipe.text_encoder.dtype)
        # pass
        # print("attention: not loading robust text encoder, using clean one instead")
    else:
        # replace text encoder with clean from hf as sanity check
        # model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        pass

    # load reconstructed texts
    with open(reconstructions_file, "r") as f:
        reconstructions = json.load(f)
    img_ids_prompts = [
        [el["original"], el["reconstructed"]] for el in reconstructions["results"]
    ]

    for i, (img_id, prompt) in enumerate(img_ids_prompts):
        print(f"img_id: {img_id} ({i}/{len(img_ids_prompts)})")
        print(f"prompt: {prompt}")
        images = pipe(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=image_length,
            width=image_length,
        ).images
        for i, image in enumerate(images):
            image.save(os.path.join(img_out_dir, f"img-{img_id}-gen-{i}.png"))
