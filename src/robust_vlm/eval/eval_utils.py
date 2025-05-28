import logging

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.hipify.hipify_python import preprocessor
from torchvision.transforms import transforms


MODEL_PATH_TO_SHORT_NAME = {
    "openai": "openai",
    "none": "original",
}

def print_statistics(arr):
    # make sure its 1-d
    assert len(arr.shape) == 1
    print(
        f"[mean] {arr.mean():.4f} [median] {np.median(arr):.4f} [min] {arr.min():.4f} [max] "
        f"{arr.max():.4f} [std] {arr.std():.4f} [n] {len(arr)}\n"
    )


def load_clip_model(clip_model_name, pretrained):
    if clip_model_name.startswith("hf-hub:"):
        # assert pretrained in (None, 'none', 'None')
        assert pretrained != "openai"
        model, _, preprocessor = open_clip.create_model_and_transforms(clip_model_name, device="cpu")
        if pretrained not in (None, "none", "None"):
            checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            model.visual.load_state_dict(checkpoint)
    elif clip_model_name.startswith("RCLIP/") or 'aimagelab' in clip_model_name or "hf_model" in clip_model_name or clip_model_name.startswith("openai/") \
        or clip_model_name.startswith("laion/"):  # bimodal robust clip project
        from transformers import AutoModel
        from PIL.Image import BICUBIC

        assert pretrained in (None, 'none', 'None')

        model = AutoModel.from_pretrained(clip_model_name)
        # todo load processor according to model
        logging.warning(f"Loading default preprocessor")
        preprocessor = transforms.Compose([  # from fare-4-clip
            transforms.Resize(224, interpolation=BICUBIC, max_size=None, antialias='warn'),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        raise ValueError(f"Unknown model name: {clip_model_name}")
    model.eval()

    # Remove the Normalize transform by creating a new Compose object
    preprocessor_no_norm = transforms.Compose(preprocessor.transforms[:-1])
    normalizer = preprocessor.transforms[-1]
    return model, preprocessor_no_norm, normalizer



@torch.inference_mode()
def compute_accuracy_no_dataloader(model, data, targets, device, batch_size=1000):
    # data, targets: tensors
    # (in parts copied from autoattack)
    train_flag = model.training
    model.eval()
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    n_total = 0
    n_correct = 0
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, data.shape[0])
        data_batch = data[start_idx:end_idx, :].clone().to(device)
        targets_batch = targets[start_idx:end_idx].clone().to(device)
        logits = model(data_batch)
        confs, preds = F.softmax(logits, dim=1).max(dim=1)
        n_total += targets_batch.size(0)
        n_correct += (preds.eq(targets_batch).sum()).item()
    acc = n_correct / n_total

    # print(f'{n_total=}')
    # print(f'{n_correct=}')
    if train_flag:
        model.train()
    return acc
