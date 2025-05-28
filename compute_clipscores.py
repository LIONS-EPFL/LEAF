import os
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from cleanfid import fid
from torchvision.datasets import CocoDetection, Flickr30k

from torchmetrics.multimodal.clip_score import CLIPScore

def is_black_image(image_path, threshold=5):
    """Check if an image is mostly black based on a pixel intensity threshold."""
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    array = np.array(image)
    if np.mean(array) < threshold:
        print(np.mean(array))
    return np.mean(array) < threshold  # Adjust threshold if needed

def filter_and_copy_images(gen_dir, original_dir, temp_gen_dir, temp_original_dir,n_samples=5000):
    """Filter out black images and copy valid images to temporary directories."""
    os.makedirs(temp_gen_dir, exist_ok=True)
    os.makedirs(temp_original_dir, exist_ok=True)

    imgs_clean = []
    imgs_gen = []
    for filename in os.listdir(original_dir):
        if filename[-3:] not in ["csv",'txt','son']:
            imgs_clean.append(filename)
    for filename in os.listdir(gen_dir):
        if filename[-3:] not in ["csv",'txt','son']:
            imgs_gen.append(filename)

    sorted_coco_images = sorted(imgs_clean, key=lambda x: int(x.split(".")[0]))
    sorted_gen_images = sorted(imgs_gen, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    num_valid = 0
    for i, (filename_real, filename_fake) in tqdm(enumerate(zip(sorted_coco_images, sorted_gen_images))):
        if i==n_samples:
            break
        gen_image_path = os.path.join(gen_dir, filename_fake)
        coco_image_path = os.path.join(original_dir, filename_real)

        if os.path.exists(coco_image_path) and not is_black_image(gen_image_path):
            shutil.copy(gen_image_path, os.path.join(temp_gen_dir, filename_fake))
            shutil.copy(coco_image_path, os.path.join(temp_original_dir, filename_real))
            num_valid += 1

    print(f"Copied {num_valid} valid images to temporary directories.")

def compute_clip(temp_gen_dir, gen_dir, original_dir, captions_path, dataset,n_samples=5000):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # Load dataset with captions
    if dataset == "coco":
        caption_dataset = CocoDetection(root=original_dir, annFile=captions_path)
    elif dataset == "flickr30k":
        print(original_dir,captions_path)
        caption_dataset = Flickr30k(root=original_dir,ann_file=captions_path)

    # load adversarial texts
    df_adv = pd.read_csv(os.path.join(gen_dir, "clip_scores.csv"))

    # Map image_id to its first caption
    image_to_caption = {}

    for i in range(len(caption_dataset)):
        _, captions = caption_dataset[i]
        if type(captions[0]) == str:
            image_to_caption[caption_dataset.ids[i]] = captions[0]
        else:
            image_to_caption[caption_dataset.ids[i]] = captions[0]['caption']
        if i<10:
            print(i, caption_dataset.ids[i], image_to_caption[caption_dataset.ids[i]])
    df = {"id":[],"clip_scores_caption_gen":[], "clip_scores_real_gen":[], "clip_scores_adv_caption_gen":[]}
    for image_path in tqdm(os.listdir(temp_gen_dir)):
        #load generated image and get index
        image = Image.open(os.path.join(temp_gen_dir,image_path))
        array = torch.tensor(np.array(image),device=device)
        id_image = caption_dataset.ids[int(image_path.split('_')[-1].split('.')[0])]

        # load captions
        caption = image_to_caption[id_image]
        adv_caption = df_adv[df_adv["id"] == id_image]['perturbed_caption'].iloc[0]
        
        #load original image
        if dataset == 'coco':
            image_original = caption_dataset._load_image(id_image)
        elif dataset == "flickr30k":
            print(os.path.join(original_dir,id_image))
            image_original = Image.open(os.path.join(original_dir,id_image))

        array_original = torch.tensor(np.array(image_original),device=device)

        with torch.no_grad():
            df["clip_scores_caption_gen"].append(metric(array,caption).item()/100)
            df["clip_scores_real_gen"].append(metric(array,array_original).item()/100)
            df["clip_scores_adv_caption_gen"].append(metric(array,adv_caption).item()/100)
        df["id"].append(id_image)
        pd.DataFrame(df).to_csv(os.path.join(gen_dir,'clip_scores_gen_img.csv'),index=False)
    return df

def compute_scores(gen_dir, original_dir,path_captions,dataset,n_samples=5000):
    """Compute FID score only on non-black images."""
    temp_gen_dir = "./temp_gen"
    temp_original_dir = "./temp_original"
    i=0
    while os.path.exists(temp_gen_dir):
        temp_gen_dir = f"./temp_gen{i}"
        temp_original_dir = f"./temp_original{i}"
        i+=1
    
    print(temp_gen_dir)

    filter_and_copy_images(gen_dir, original_dir, temp_gen_dir, temp_original_dir,n_samples=n_samples)

    # Compute FID on the filtered image directories
    #fid_score = fid.compute_fid(temp_gen_dir, temp_original_dir)
    df = compute_clip(temp_gen_dir, gen_dir = gen_dir, original_dir = original_dir, captions_path = path_captions, dataset=dataset,n_samples=n_samples)

    # Clean up temporary directories
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_original_dir)
    #print(f"FID Score: {fid_score}")
    #return fid_score
    return 1000

if __name__ == '__main__':
    # Paths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="coco"
        )
    parser.add_argument(
        "--coco_real", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--flickr30k_real", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--path_fake", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--coco_captions", 
        type=str, 
        default=None
        )
    
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=5000
        )
    
    parser.add_argument(
        "--flickr30k_captions", 
        type=str, 
        default=None
        )

    args = parser.parse_args()

    if args.dataset == "coco":
        path_captions = args.coco_captions
        path_real = args.coco_real
    elif args.dataset == "flickr30k":
        path_captions = args.flickr30k_captions
        path_real = args.flickr30k_real

    fid = compute_scores(args.path_fake, path_real, path_captions, args.dataset, n_samples = args.n_samples)

    df = {'fid':[fid]}
    pd.DataFrame(df).to_csv(os.path.join(args.path_fake,'generation_metrics.csv'),index=False)
