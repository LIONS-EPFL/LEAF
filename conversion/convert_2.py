# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Found in https://gist.github.com/rwightman/c79fd0241ed3c860e898114931c07990
'''

import argparse
import os.path

import torch
import numpy as np

# otherwise raises exception...
torch.serialization.add_safe_globals([
    np.dtype, 
    np.core.multiarray.scalar, 
    np.float64,  # Explicitly adding float64 dtype
    np.dtypes.Float64DType  # Newly blocked type
])
from open_clip import create_model
from transformers import CLIPConfig, CLIPVisionConfig, CLIPTextConfig, CLIPModel


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    assert(hf_attn_layer.num_heads == pt_attn_layer.num_heads)
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    hf_attn_layer.q_proj.weight.copy_(q_proj)
    hf_attn_layer.q_proj.bias.copy_(q_proj_bias)

    hf_attn_layer.k_proj.weight.copy_(k_proj)
    hf_attn_layer.k_proj.bias.copy_(k_proj_bias)

    hf_attn_layer.v_proj.weight.copy_(v_proj)
    hf_attn_layer.v_proj.bias.copy_(v_proj_bias)

    hf_attn_layer.out_proj.weight.copy_(pt_attn_layer.out_proj.weight)
    hf_attn_layer.out_proj.bias.copy_(pt_attn_layer.out_proj.bias)


def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    hf_linear.weight.copy_(pt_linear.weight)
    hf_linear.bias.copy_(pt_linear.bias)


def copy_layer(hf_layer, pt_layer):
    # copy layer norms
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # copy MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # copy attn
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    # copy  embeds
    hf_encoder.embeddings.token_embedding.weight.copy_(pt_model.token_embedding.weight)
    hf_encoder.embeddings.position_embedding.weight.copy_(pt_model.positional_embedding)

    # copy layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # copy hidden layers
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.text_projection.weight.copy_(pt_model.text_projection.T)

    # copy text encoder
    copy_encoder(hf_model.text_model, pt_model)


def copy_vison_model_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.visual_projection.weight.copy_(pt_model.visual.proj.T)

    # copy layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # copy embeds
    hf_model.vision_model.embeddings.patch_embedding.weight.copy_(pt_model.visual.conv1.weight)
    hf_model.vision_model.embeddings.class_embedding.copy_(pt_model.visual.class_embedding)
    hf_model.vision_model.embeddings.position_embedding.weight.copy_(pt_model.visual.positional_embedding)

    # copy encoder
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


@torch.no_grad()
def convert_clip_checkpoint(model, pretrained, model_name=None,image_encoder_only=False):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    #CLIPVisionConfig()
    #CLIPTextConfig()

    #L14
    if model_name == 'L':
        config = CLIPConfig(
            projection_dim=768,
            text_config_dict=dict(
                hidden_act='quick_gelu',
                #hidden_act='gelu',
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
            ),
            vision_config_dict=dict(
                hidden_act='quick_gelu',
                #hidden_act='gelu',
                num_hidden_layers=24,
                patch_size=14,
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
            ))

    ## H14
    elif model_name == 'H':
        config = CLIPConfig(
            projection_dim=1024,
            text_config_dict=dict(
                hidden_act='gelu',
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
            ),
            vision_config_dict=dict(
                hidden_act='gelu',
                num_hidden_layers=32,
                patch_size=14,
                hidden_size=1280,
                intermediate_size=5120,
                num_attention_heads=16,
            ))
    ## B16 / B16 plus
    elif model_name == 'B':
        config = CLIPConfig(
            projection_dim=512,
            text_config_dict=dict(
                hidden_act='quick_gelu',
            ),
            vision_config_dict=dict(
                hidden_act='quick_gelu',
                num_hidden_layers=12,
                patch_size=32
            ))
    # ## g14
    elif model_name=='g':
        config = CLIPConfig(
            projection_dim=1024,
            text_config_dict=dict(
                hidden_act='gelu',
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
            ),
            vision_config_dict=dict(
                hidden_act='gelu',
                num_hidden_layers=40,
                patch_size=14,
                hidden_size=1408,
                intermediate_size=6144,
                num_attention_heads=16,
            ))
    elif model_name=='G':
        config = CLIPConfig(
            projection_dim=1280,  # Typically 1024 for CLIP models
            text_config_dict=dict(
                hidden_act='gelu',
                hidden_size=1280,  # CLIP text encoder for bigG typically uses 1280
                intermediate_size=5120,
                num_attention_heads=20,  # Adjusted to match the text encoder scale
                num_hidden_layers=32,  # Bigger than standard ViT-L
            ),
            vision_config_dict=dict(
                hidden_act='gelu',
                num_hidden_layers=48,  # ViT-bigG has 48 transformer layers
                patch_size=14,  # Consistent with ViT-L/14 and ViT-H/14
                hidden_size=1664,  # Hidden size for ViT-bigG
                intermediate_size=8192,  # Typically 4x hidden_size
                num_attention_heads=16,  # Heads stay the same as ViT-H
            ))

    print(config)
    hf_model = CLIPModel(config).eval()
    print(hf_model)

    if image_encoder_only:
        pt_model = create_model(
            model, precision='amp', force_quick_gelu=(model_name in ["B", "L"])
            )
        sd = torch.load(pretrained, map_location='cpu')
        pt_model.visual.load_state_dict(sd)
    else:
        pt_model = create_model(model, pretrained=pretrained, precision='amp', force_quick_gelu=(model_name in ["B","L"]))
    pt_model = pt_model.eval()
    print(pt_model)

    copy_text_model_and_projection(hf_model, pt_model)
    copy_vison_model_and_projection(hf_model, pt_model)
    hf_model.logit_scale = pt_model.logit_scale

    input_ids = torch.tensor([49406] + list(torch.arange(1, 77, dtype=int))).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)

    hf_image_embed = hf_model.get_image_features(pixel_values)
    hf_text_embed = hf_model.get_text_features(input_ids)

    pt_image_embed = pt_model.encode_image(pixel_values)
    pt_text_embed = pt_model.encode_text(input_ids)

    print(pt_text_embed.sum())
    print(hf_text_embed.sum())

    print((pt_image_embed - hf_image_embed).sum())
    print((pt_text_embed - hf_text_embed).sum())
    print((pt_text_embed - hf_text_embed).max(), (pt_text_embed - hf_text_embed).min())
    assert torch.allclose(hf_image_embed, pt_image_embed, atol=1e-4)
    assert torch.allclose(hf_text_embed, pt_text_embed, atol=1e-4)


    hf_logits_per_image, hf_logits_per_text = hf_model(
        input_ids=input_ids, pixel_values=pixel_values, return_dict=False
    )[:2]

    pt_image_features, pt_text_features, logit_scale = pt_model(pixel_values, input_ids)
    pt_logits_per_image = pt_image_features @ pt_text_features.T * logit_scale
    pt_logits_per_text = pt_logits_per_image.T

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-4)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-4)


    out_folder = os.path.abspath(os.path.join(pretrained, os.pardir))
    print('out_folder:',out_folder)

    if os.path.exists(pretrained):
        pretrained = os.path.splitext(os.path.basename(pretrained))[0]

    hf_model.save_pretrained(os.path.join(out_folder,'hf_model'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--pretrained", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--model_name", default=None, type=str, help="Letter of the ViT trying to convert, e.g., L for ViT-L")
    args = parser.parse_args()

    convert_clip_checkpoint(args.model, args.pretrained, args.model_name)
