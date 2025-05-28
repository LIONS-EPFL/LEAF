import sys

if len(sys.argv) < 2:
    sys.exit(
        """Usage: python run.py path-to-image [path-to-image-2 ...]
        Passing multiple images will optimize a single prompt across all passed images, useful for style transfer.
        """
        )

# config_path = "sample_config.json"
config_path = "my_config.json"

# image_paths = sys.argv[1:]
# load the target image
# images = [Image.open(image_path) for image_path in image_paths]

text = sys.argv[1:]
print(f"Text: {text}")
assert len(text) == 1

# defer loading other stuff until we confirm the images loaded
import argparse
from optim_utils import *

print("Initializing...")

# load args
args = argparse.Namespace()
args.__dict__.update(read_json(config_path))

# You may modify the hyperparamters here
args.print_new_best = True
# args.clip_pretrain = ("/mnt/cschlarmann37/project_fuse-clip/openclip-checkpoints/"
#                       "OpenCLIP-ViT-H-rho50-k1-constrained-FARE2.pt")
if args.prompt_len == "match":
    print("Warning: using default tokenizer")
    ids = open_clip_pez.tokenize(text).squeeze()
    args.prompt_len = len(ids[ids!=0]) - 2  # bot and eot tokens are prepended/appended by dummy ids later

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip_pez.create_model_and_transforms(
    args.clip_model, pretrained=args.clip_pretrain, device=device
    )

print(f"Running for {args.iter} steps.")
if getattr(args, 'print_new_best', False) and args.print_step is not None:
    print(f"Intermediate results will be printed every {args.print_step} steps.")

# optimize prompt
learned_prompt = optimize_prompt(
    model, preprocess, args, device,
    # target_images=images
    target_prompts=text,
)
print(learned_prompt)
