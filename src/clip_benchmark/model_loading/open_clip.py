import sys
import open_clip
import torch
from torchvision import transforms


def load_open_clip(
    model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"
):
    if model_name.startswith("hf-hub:"):
        # assert pretrained in (None, 'none', 'None')
        assert pretrained != "openai"
        model, _, transform = open_clip.create_model_and_transforms(model_name, device="cpu")
        if pretrained not in (None, "none", "None"):
            checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            model.visual.load_state_dict(checkpoint)

        if "convnext" in model_name.lower():
            transform.transforms[1] = transforms.CenterCrop(224)  # todo
            print("Using CenterCrop(224) for convnext")
    else:
        try:
            model, _, transform = open_clip.create_model_and_transforms(
                model_name, pretrained="openai", cache_dir=cache_dir, device="cpu"
            )
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            else:
                checkpoint = pretrained
            model.visual.load_state_dict(checkpoint)
        except Exception as e:
            # try loading whole model
            print(f"error: {e}", file=sys.stderr)
            print("retrying by loading whole model..", file=sys.stderr)
            torch.cuda.empty_cache()
            model, _, transform = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, cache_dir=cache_dir, device="cpu"
            )

    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
