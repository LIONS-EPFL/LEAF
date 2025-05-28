import logging
import sys
import open_clip
import torch
from torchvision import transforms
from robust_vlm.eval.eval_utils import load_clip_model

def load_hf_clip(model_name: str, pretrained: str, cache_dir: str, device: str):
    assert pretrained in [None, "none"], pretrained
    assert cache_dir in [None, "none"], cache_dir

    model, preprocessor_no_norm, normalizer = load_clip_model(clip_model_name=model_name, pretrained=pretrained)
    preprocessor = transforms.Compose([preprocessor_no_norm, normalizer])  # gets split later
    tokenizer = open_clip.tokenize
    logging.warning("using default tokenizer")

    model = model.to(device)
    return model, preprocessor, tokenizer