import os
import types
import torch
import argparse
import pandas as pd
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, Flickr30k
from diffusers import AutoPipelineForText2Image
from utils_attacks import attack_text_charmer_inference, convert_clip_text_model, encode_text_wrapper, encode_text_wrapper_2, tokenizer_wrapper
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTextModel


# Create a dataset with one caption per image
class CocoSingleCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.img_ids = dataset.ids
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image, captions = self.dataset[idx]  # Get image
        #print(captions)
        if type(captions[0]) == str:
            return image, captions[0]
        else:
            return image, captions[0]['caption']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
        type=str,
        default="coco"
    )
    parser.add_argument(
        "--coco_root", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--flickr30k_root", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="stable-diffusion-v1-5/stable-diffusion-v1-5"
        )
    parser.add_argument(
        "--adv",
        action="store_true",
        default=False,
        help="Perform adversarial attacks to the text before the image generation process",
        )
    parser.add_argument(
        "--constrain",
        action="store_true",
        default=False,
        help="Constrain the adversarial attacks to not produce words with meaning",
        )
    parser.add_argument(
        "--adv_objective", 
        type=str, 
        default="dissim"
        )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=-1
        )
    parser.add_argument(
        "--rho", 
        type=int, 
        default=20,
        help="number of candidate positions to consider with Charmer"
        )
    parser.add_argument(
        "--k", 
        type=int, 
        default=1,
        help="maximum edit distance"
        )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=20
        )
    parser.add_argument(
        "--num_steps", 
        type=int, 
        default=50
        )
    parser.add_argument(
        "--text_encoder_name", 
        type=str, default=None, 
        help="Name of the text encoder to use, if none, the default one from the diffusion model will be used"
        )

    parser.add_argument(
        "--text_encoder_name_2", 
        type=str, default=None, 
        help="Name of the second text encoder to use, if none, the default one from the diffusion model will be used"
        )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "coco":

        # Paths to COCO dataset
        dataset_root = args.coco_root
        ann_file = f"{dataset_root}/annotations/captions_val2017.json"
        img_folder = f"{dataset_root}/images"

    elif args.dataset == "flickr30k":

        # Paths to flickr30 dataset
        dataset_root = args.flickr30k_root
        ann_file = f"{dataset_root}/clean_captions.txt"
        img_folder = f"{dataset_root}/images"
    
    output_dir = os.path.join(dataset_root,"generated_images")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    out_folder = os.path.join(output_dir, (f"Adv_k{args.k}_rho{args.rho}_" if args.adv else "") +
                                   (args.adv_objective + "_" if args.adv_objective!="dissim" else "") + 
                                   (f"constrained_" if args.constrain else "") + 
                                   args.model_name.split("/")[-1] + f"_{args.num_steps}steps" + 
                                   (f"_text_encoder_{args.text_encoder_name.split('/')[-1]}" if args.text_encoder_name is not None else "") + 
                                   (f"_text_encoder_2_{args.text_encoder_name_2.split('/')[-1]}" if args.text_encoder_name_2 is not None else ""))
    os.makedirs(out_folder, exist_ok=True)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.ToTensor(),          # Convert to PyTorch tensor
    ])

    caption_ann_file = ann_file

    # Load dataset with captions
    if args.dataset == "coco":
        caption_dataset = CocoDetection(root=img_folder, annFile=caption_ann_file, transform=transform)

    elif args.dataset == "flickr30k":
        caption_dataset = Flickr30k(root = img_folder, ann_file = caption_ann_file, transform=transform)
    
    single_caption_dataset = CocoSingleCaptionDataset(caption_dataset)


    dataloader = DataLoader(single_caption_dataset, batch_size=args.batch_size, shuffle=False)

    # Test loading one sample
    image, captions = caption_dataset[0]
    print("Image shape:", image.shape)
    print("Captions:",captions)
    # print("Captions:", [cap["caption"] for cap in captions])

    # Initialize pipeline
    pipeline = AutoPipelineForText2Image.from_pretrained(args.model_name, torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32))
    pipeline = pipeline.to(device)

    # Replace text encoder if specified
    if args.text_encoder_name is not None:
        model = CLIPTextModel.from_pretrained(args.text_encoder_name)

        pipeline.text_encoder = model.to(device,dtype=pipeline.text_encoder.dtype)
    if args.text_encoder_name_2 is not None:
        model_2 = convert_clip_text_model(pipeline.text_encoder_2, CLIPTextModel.from_pretrained(args.text_encoder_name_2)).to(device)
        pipeline.text_encoder_2 = model_2.to(device,dtype=pipeline.text_encoder_2.dtype)

    # initialize CLIP
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    with torch.no_grad():
        df = {"id": [], "original_caption": [], "perturbed_caption": [], "caps_corr_model_text_encoder":[],"caps_clip_score": [], "og_img_clip_score":[], "pt_img_clip_score":[]}
        for i,batch in tqdm(enumerate(dataloader)):
            images, captions = batch
            #print(captions)
            if i==args.num_samples:
                break

            perturbed_captions = []
            for j in range(len(captions)):
                df["id"].append(caption_dataset.ids[i*args.batch_size + j])
                df["original_caption"].append(captions[j])

                # Needed for CLIPScore
                image = (images[j] * 255).byte().to(device)

                if args.adv:
                    # Wrap stuff
                    tokenizer = tokenizer_wrapper(pipeline.tokenizer)
                    pipeline.text_encoder.encode_text = types.MethodType(encode_text_wrapper, pipeline.text_encoder)
                    if args.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
                        pipeline.text_encoder_2.encode_text = types.MethodType(encode_text_wrapper_2, pipeline.text_encoder_2)

                    # compute original features
                    tokens = tokenizer(captions[j]).unsqueeze(0).to(device)
                    text_features_frozen = pipeline.text_encoder(tokens).pooler_output

                    if args.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
                        text_features_frozen_2 = pipeline.text_encoder_2(tokens).text_embeds
                        adv_text, _ = attack_text_charmer_inference(pipeline.text_encoder,tokenizer,captions[j],text_features_frozen,device,objective=args.adv_objective,n=args.rho,k=args.k,constrain=args.constrain,debug=True,model_2=pipeline.text_encoder_2, model_2_anchor_features=text_features_frozen_2, batch_size=256) 
                        tokens_adv = tokenizer(adv_text).unsqueeze(0).to(device)
                        text_features_adv = pipeline.text_encoder(tokens_adv).pooler_output
                        text_features_adv_2 = pipeline.text_encoder_2(tokens_adv).text_embeds
                        corr = text_features_frozen@text_features_adv.T/(torch.norm(text_features_adv) * torch.norm(text_features_frozen))       
                        corr += text_features_frozen_2@text_features_adv_2.T/(torch.norm(text_features_adv_2) * torch.norm(text_features_frozen_2))
                        corr /= 2
                    else:   
                        adv_text, _ = attack_text_charmer_inference(pipeline.text_encoder,tokenizer,captions[j],text_features_frozen,device,objective=args.adv_objective,n=args.rho,k=args.k,constrain=args.constrain,debug=True, batch_size=512)        
                        tokens_adv = tokenizer(adv_text).unsqueeze(0).to(device)
                        text_features_adv = pipeline.text_encoder(tokens_adv).pooler_output
                        corr = text_features_frozen@text_features_adv.T/(torch.norm(text_features_adv) * torch.norm(text_features_frozen))
                    
                    perturbed_captions.append(adv_text)
                    df["perturbed_caption"].append(adv_text)
                    df["caps_corr_model_text_encoder"].append(corr.item())
                    df["caps_clip_score"].append(metric(captions[j], adv_text).item()/100)
                    df['og_img_clip_score'].append(metric(image,captions[j]).item()/100)
                    df['pt_img_clip_score'].append(metric(image,adv_text).item()/100)
                else:
                    perturbed_captions.append(captions[j])
                    df["perturbed_caption"].append(captions[j])
                    df["caps_corr_model_text_encoder"].append(None)
                    df["caps_clip_score"].append(None)
                    df['og_img_clip_score'].append(metric(image,captions[j]).item()/100)
                    df['pt_img_clip_score'].append(None)
                pd.DataFrame(df).to_csv(os.path.join(out_folder,'clip_scores.csv'),index=False)

            generated_images = pipeline(perturbed_captions, num_inference_steps=args.num_steps).images
            for j, image in enumerate(generated_images):
                image.save(os.path.join(out_folder, f"generated_image_{i*args.batch_size + j}.png"))

