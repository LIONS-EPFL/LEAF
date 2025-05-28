import os
import sys
import argparse

from tqdm import tqdm

from optim_utils import *


def run_one_inversion(text, args):
    print(f"Text: {text}")
    assert len(text) == 1

    print("Initializing...")
    # copy the args
    args = copy.deepcopy(args)
    if args.prompt_len == "match":
        print("Warning: using default tokenizer")
        ids = open_clip_pez.tokenize(text).squeeze()
        args.prompt_len = len(ids[ids != 0]) - 2  # bot and eot tokens are prepended/appended by dummy ids later
        print(f"Matching prompt length to {args.prompt_len}.")

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Running for {args.iter} steps.")
    if getattr(args, 'print_new_best', False) and args.print_step is not None:
        print(f"Intermediate results will be printed every {args.print_step} steps.")

    res_dict = {"original": text[0]}
    # optimize prompt
    res_dict_ = optimize_prompt(
        model, preprocess, args, device,
        # target_images=images
        target_prompts=text,
    )
    res_dict.update(res_dict_)
    return res_dict


def get_captions():
    coco_captions_path = "/mnt/datasets/coco/annotations/captions_val2017.json"

    captions = json.load(open(coco_captions_path, 'r'))['annotations']
    # get captions for unique images
    img_ids = list()
    our_captions = list()
    for el in captions:
        if el['image_id'] not in img_ids:
            img_ids.append(el['image_id'])
            our_captions.append(el['caption'])
    print("\n".join(our_captions[:100]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='vit-h-14'
    )
    parser.add_argument('--robust', action='store_true', help="Use robust model")
    parser.add_argument('--iter', type=int, default=3000, help="Number of iterations")
    parser.add_argument('--n-samples', type=int, default=100, help="Number of samples to run")

    config_path = "coco_config.json"
    our_captions_path = "coco_captions.txt"

    MODEL_NAME_DICT = { # ["model_name", "pretrained_clean", "pretrained_robust"]
        "vit-l-14": ["ViT-L-14-quickgelu",
                     "openai",
                     "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     "CLIP-ViT-L-rho50-k1-constrained.pt"],
        "vit-h-14": ["ViT-H-14",
                     "laion2b_s32b_b79k",
                     "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     "OpenCLIP-ViT-H-rho50-k1-constrained-FARE2.pt"],
        "vit-g-14": ["ViT-g-14",
                     "laion2b_s12b_b42k",
                     "/mnt/cschlarmann37/project_bimodal-robust-clip/openclip-checkpoints/"
                     "OpenCLIP-ViT-g-rho50-k1-constrained-FARE2.pt"],
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
    with open(our_captions_path, 'r') as f:
        our_captions = f.readlines()[:args.n_samples]

    # load CLIP model
    pretrained = MODEL_NAME_DICT[args.model][2] if args.robust else MODEL_NAME_DICT[args.model][1]
    args.model = MODEL_NAME_DICT[args.model][0]
    print(f"Loading {args.model} with pretrained {pretrained}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip_pez.create_model_and_transforms(
        args.model, pretrained=pretrained,
        device=device
    )

    results = []
    for i, caption in enumerate(tqdm(our_captions)):
        print(f"{i} / {len(our_captions)}")
        our_captions[i] = caption.strip()
        res_dict = run_one_inversion([our_captions[i]], args=args)
        print(f"reconstructed: {res_dict['reconstructed']}")
        print(f"original: {our_captions[i]}")

        results.append(res_dict)

    results = {
        "config": args.__dict__,
        "results": results
    }
    # save results
    results_file = (f"results-{args.n_samples}smpls-{args.iter}iters-{args.model}"
                    f"-{'robust' if args.robust else 'clean'}.json")
    results_path = os.path.join("/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions", results_file)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)







