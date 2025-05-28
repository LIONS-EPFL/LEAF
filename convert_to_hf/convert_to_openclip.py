  #!/usr/bin/env python3
"""
convert_hf_to_openclip.py

Convert a Hugging Face Transformers CLIP checkpoint to OpenAI-/OpenCLIP-style state dict,
with sanity checks added.
"""
import os
import argparse
import torch
from transformers import CLIPModel
import open_clip

def main(args):
    print(f"Loading HF CLIP model from {args.hf_checkpoint}...")
    hf_model = CLIPModel.from_pretrained(args.hf_checkpoint)
    hf_state = hf_model.state_dict()

    print(f"Instantiating empty OpenCLIP '{args.openclip_model}' for shape reference...")
    oc_model_ref, _, _ = open_clip.create_model_and_transforms(
        args.openclip_model, pretrained=args.pretrained
    )

    oc_state = {}

    # ---- Vision embeddings & projection ----
    oc_state['visual.conv1.weight'] = hf_state['vision_model.embeddings.patch_embedding.weight']
    oc_state['visual.class_embedding'] = hf_state['vision_model.embeddings.class_embedding']
    oc_state['visual.positional_embedding'] = hf_state['vision_model.embeddings.position_embedding.weight']
    oc_state['visual.ln_pre.weight'] = hf_state['vision_model.pre_layrnorm.weight']
    oc_state['visual.ln_pre.bias']  = hf_state['vision_model.pre_layrnorm.bias']

    # Transformer blocks (vision)
    v_layers = max(int(k.split('.')[3]) for k in hf_state
                   if k.startswith('vision_model.encoder.layers') and 'q_proj.weight' in k) + 1
    for i in range(v_layers):
        # Q, K, V → in_proj
        q = hf_state[f'vision_model.encoder.layers.{i}.self_attn.q_proj.weight']
        k = hf_state[f'vision_model.encoder.layers.{i}.self_attn.k_proj.weight']
        v = hf_state[f'vision_model.encoder.layers.{i}.self_attn.v_proj.weight']
        oc_state[f'visual.transformer.resblocks.{i}.attn.in_proj_weight'] = torch.cat([q, k, v], dim=0)
        qb = hf_state[f'vision_model.encoder.layers.{i}.self_attn.q_proj.bias']
        kb = hf_state[f'vision_model.encoder.layers.{i}.self_attn.k_proj.bias']
        vb = hf_state[f'vision_model.encoder.layers.{i}.self_attn.v_proj.bias']
        oc_state[f'visual.transformer.resblocks.{i}.attn.in_proj_bias']   = torch.cat([qb, kb, vb], dim=0)

        # out_proj
        oc_state[f'visual.transformer.resblocks.{i}.attn.out_proj.weight'] = \
            hf_state[f'vision_model.encoder.layers.{i}.self_attn.out_proj.weight']
        oc_state[f'visual.transformer.resblocks.{i}.attn.out_proj.bias']   = \
            hf_state[f'vision_model.encoder.layers.{i}.self_attn.out_proj.bias']

        # MLP
        oc_state[f'visual.transformer.resblocks.{i}.mlp.c_fc.weight'] = \
            hf_state[f'vision_model.encoder.layers.{i}.mlp.fc1.weight']
        oc_state[f'visual.transformer.resblocks.{i}.mlp.c_fc.bias']   = \
            hf_state[f'vision_model.encoder.layers.{i}.mlp.fc1.bias']
        oc_state[f'visual.transformer.resblocks.{i}.mlp.c_proj.weight'] = \
            hf_state[f'vision_model.encoder.layers.{i}.mlp.fc2.weight']
        oc_state[f'visual.transformer.resblocks.{i}.mlp.c_proj.bias']   = \
            hf_state[f'vision_model.encoder.layers.{i}.mlp.fc2.bias']

        # Norms
        oc_state[f'visual.transformer.resblocks.{i}.ln_1.weight'] = \
            hf_state[f'vision_model.encoder.layers.{i}.layer_norm1.weight']
        oc_state[f'visual.transformer.resblocks.{i}.ln_1.bias']   = \
            hf_state[f'vision_model.encoder.layers.{i}.layer_norm1.bias']
        oc_state[f'visual.transformer.resblocks.{i}.ln_2.weight'] = \
            hf_state[f'vision_model.encoder.layers.{i}.layer_norm2.weight']
        oc_state[f'visual.transformer.resblocks.{i}.ln_2.bias']   = \
            hf_state[f'vision_model.encoder.layers.{i}.layer_norm2.bias']

    # Final vision LN & projection
    oc_state['visual.ln_post.weight'] = hf_state['vision_model.post_layernorm.weight']
    oc_state['visual.ln_post.bias']   = hf_state['vision_model.post_layernorm.bias']
    oc_state['visual.proj'] = hf_state['visual_projection.weight'].T

    # ---- Text embeddings & projection ----
    oc_state['token_embedding.weight']   = hf_state['text_model.embeddings.token_embedding.weight']
    oc_state['positional_embedding']     = hf_state['text_model.embeddings.position_embedding.weight']

    t_layers = max(int(k.split('.')[3]) for k in hf_state
                   if k.startswith('text_model.encoder.layers') and 'q_proj.weight' in k) + 1
    for i in range(t_layers):
        q = hf_state[f'text_model.encoder.layers.{i}.self_attn.q_proj.weight']
        k = hf_state[f'text_model.encoder.layers.{i}.self_attn.k_proj.weight']
        v = hf_state[f'text_model.encoder.layers.{i}.self_attn.v_proj.weight']
        oc_state[f'transformer.resblocks.{i}.attn.in_proj_weight'] = torch.cat([q, k, v], dim=0)
        qb = hf_state[f'text_model.encoder.layers.{i}.self_attn.q_proj.bias']
        kb = hf_state[f'text_model.encoder.layers.{i}.self_attn.k_proj.bias']
        vb = hf_state[f'text_model.encoder.layers.{i}.self_attn.v_proj.bias']
        oc_state[f'transformer.resblocks.{i}.attn.in_proj_bias']   = torch.cat([qb, kb, vb], dim=0)

        oc_state[f'transformer.resblocks.{i}.attn.out_proj.weight'] = \
            hf_state[f'text_model.encoder.layers.{i}.self_attn.out_proj.weight']
        oc_state[f'transformer.resblocks.{i}.attn.out_proj.bias']   = \
            hf_state[f'text_model.encoder.layers.{i}.self_attn.out_proj.bias']

        oc_state[f'transformer.resblocks.{i}.mlp.c_fc.weight'] = \
            hf_state[f'text_model.encoder.layers.{i}.mlp.fc1.weight']
        oc_state[f'transformer.resblocks.{i}.mlp.c_fc.bias']   = \
            hf_state[f'text_model.encoder.layers.{i}.mlp.fc1.bias']
        oc_state[f'transformer.resblocks.{i}.mlp.c_proj.weight'] = \
            hf_state[f'text_model.encoder.layers.{i}.mlp.fc2.weight']
        oc_state[f'transformer.resblocks.{i}.mlp.c_proj.bias']   = \
            hf_state[f'text_model.encoder.layers.{i}.mlp.fc2.bias']

        oc_state[f'transformer.resblocks.{i}.ln_1.weight'] = \
            hf_state[f'text_model.encoder.layers.{i}.layer_norm1.weight']
        oc_state[f'transformer.resblocks.{i}.ln_1.bias']   = \
            hf_state[f'text_model.encoder.layers.{i}.layer_norm1.bias']
        oc_state[f'transformer.resblocks.{i}.ln_2.weight'] = \
            hf_state[f'text_model.encoder.layers.{i}.layer_norm2.weight']
        oc_state[f'transformer.resblocks.{i}.ln_2.bias']   = \
            hf_state[f'text_model.encoder.layers.{i}.layer_norm2.bias']

    # Final text LN & projection
    oc_state['ln_final.weight']   = hf_state['text_model.final_layer_norm.weight']
    oc_state['ln_final.bias']     = hf_state['text_model.final_layer_norm.bias']
    oc_state['text_projection']   = hf_state['text_projection.weight'].T
    oc_state['logit_scale']       = hf_state['logit_scale']

    # ---- Sanity Checks ----
    print("Performing sanity checks...")
    oc_model_converted, _, _ = open_clip.create_model_and_transforms(
        args.openclip_model, pretrained=args.pretrained
    )
    sd_res = oc_model_converted.load_state_dict(oc_state, strict=False)
    if sd_res.missing_keys or sd_res.unexpected_keys:
        print("WARNING: mismatch in state dict load:")
        if sd_res.missing_keys:
            print("  Missing keys:", sd_res.missing_keys)
        if sd_res.unexpected_keys:
            print("  Unexpected keys:", sd_res.unexpected_keys)
    else:
        print("All keys matched successfully.")

    # Forward-pass test with random inputs
    print("Running forward-pass test...")
    input_ids = torch.tensor([49406] + list(torch.arange(1, 77, dtype=int))).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)

    hf_image_embed = hf_model.get_image_features(pixel_values)
    hf_text_embed = hf_model.get_text_features(input_ids)

    oc_image_embed = oc_model_converted.encode_image(pixel_values)
    oc_text_embed = oc_model_converted.encode_text(input_ids)

    print("sum oc ", oc_text_embed.sum())
    print("sum hf ",  hf_text_embed.sum())

    print("image diff sum ", (oc_image_embed - hf_image_embed).sum())
    print("text diff sum ", (oc_text_embed - hf_text_embed).sum())
    print("max ", (oc_text_embed - hf_text_embed).max(), "min ", (oc_text_embed - hf_text_embed).min())
    assert torch.allclose(hf_image_embed, oc_image_embed, atol=1e-4)
    assert torch.allclose(hf_text_embed, oc_text_embed, atol=1e-4)

    hf_logits_per_image, hf_logits_per_text = hf_model(
        input_ids=input_ids, pixel_values=pixel_values, return_dict=False
    )[:2]

    pt_image_features, pt_text_features, logit_scale = oc_model_converted(pixel_values, input_ids)
    pt_logits_per_image = pt_image_features @ pt_text_features.T * logit_scale
    pt_logits_per_text = pt_logits_per_image.T

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-4)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-4)
    print("Forward-pass sanity check passed.")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print(f"Saving converted state dict to {args.output}")
    torch.save(oc_state, args.output)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HF Transformers CLIP checkpoint to OpenCLIP (OpenAI) format"
    )
    parser.add_argument(
        "--hf_checkpoint", type=str,
        default="",
        help="HF model name or local path (e.g. openai/clip-vit-base-patch32)"
    )
    parser.add_argument(
        "--openclip_model", type=str,
        default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="OpenCLIP architecture name (e.g. 'ViT-B-32')"
    )
    parser.add_argument(
        "--output", type=str,
        default="",
        help="Output path for converted state dict (e.g. open_clip_pytorch_model.bin)"
    )
    args = parser.parse_args()

    args.hf_checkpoint = ""
    args.openclip_model = "ViT-L-14-quickgelu"
    args.pretrained = None  # "openai"
    args.output = f"" \
                  f"{args.hf_checkpoint.split('/')[1]}.pt"

    main(args)
