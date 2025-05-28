import os
import sys
import argparse

from tqdm import tqdm

from optim_utils import *


def run_one_inversion(image, args):
    image_id, image = image

    print("Initializing...")
    # copy the args
    args = copy.deepcopy(args)
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Running for {args.iter} steps.")
    if getattr(args, 'print_new_best', False) and args.print_step is not None:
        print(f"Intermediate results will be printed every {args.print_step} steps.")

    res_dict = {"original": image_id}
    # optimize prompt
    res_dict_ = optimize_prompt(
        model, preprocess, args, device,
        target_images=[image]
        # target_prompts=text,
    )
    res_dict.update(res_dict_)
    return res_dict


def get_images(n=100):
    # load n random images
    coco_images_path = "/mnt/datasets/coco/val2017/"
    rng = np.random.default_rng(0)
    image_ids = rng.choice(os.listdir(coco_images_path), n, replace=False)
    images = {}
    for image_id in image_ids:
        image_path = os.path.join(coco_images_path, image_id)
        image = Image.open(image_path).convert("RGB")
        images[image_id] = image
    return images



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='vit-h-14'
    )
    parser.add_argument('--robust', action='store_true', help="Use robust model")
    parser.add_argument('--iter', type=int, default=3000, help="Number of iterations")
    parser.add_argument('--n-samples', type=int, default=100, help="Number of samples to run")

    config_path = "coco_img_config.json"

    MODEL_NAME_DICT = { # ["model_name", "pretrained_clean", "pretrained_robust"]
        "vit-l-14": ["ViT-L-14-quickgelu",
                     "openai",
                     "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     "CLIP-ViT-L-rho50-k1-constrained.pt"  # using clean image encoder
                     ],
        "vit-h-14": ["ViT-H-14",
                     "laion2b_s32b_b79k",
                     # "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     # "OpenCLIP-ViT-H-rho50-k1-constrained-FARE2.pt"
                     "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     "OpenCLIP-ViT-H-rho50-k1-constrained.pt"  # using clean image encoder
                     ],
        "vit-g-14": ["ViT-g-14",
                     "laion2b_s12b_b42k",
                     # "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     # "OpenCLIP-ViT-g-rho50-k1-constrained-FARE2.pt"
                     ],
        "vit-bigG-14": ["ViT-bigG-14",
                        "laion2b_s39b_b160k",
                        "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                        "OpenCLIP-ViT-bigG-rho50-k1-constrained.pt"]
    }

    # load args
    args = argparse.Namespace()
    args.__dict__.update(read_json(config_path))
    args.__dict__.update(vars(parser.parse_args()))

    # You may modify the hyperparamters here
    args.print_new_best = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"running on {args.n_samples} samples")
    images = get_images(args.n_samples)

    # load CLIP model
    pretrained = MODEL_NAME_DICT[args.model][2] if args.robust else MODEL_NAME_DICT[args.model][1]
    args.model = MODEL_NAME_DICT[args.model][0]
    print(f"Loading {args.model} with pretrained {pretrained}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip_pez.create_model_and_transforms(
        args.model, pretrained=pretrained, device=device
    )

    results = []
    for i, (image_id, image) in enumerate(tqdm(images.items())):
        print(f"{i} / {len(images)}")
        res_dict = run_one_inversion([image_id, image], args=args)
        print(f"reconstructed: {res_dict['reconstructed']}")
        print(f"original: {image_id}")

        results.append(res_dict)

    results = {
        "config": args.__dict__,
        "results": results
    }
    # save results
    results_file = (f"results-coco-img-{args.n_samples}smpls-{args.iter}iters-{args.model}"
                    f"-{'robust' if args.robust else 'clean'}.json")
    results_path = os.path.join(
        f"/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions/coco-img-{args.model}",
        results_file
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

