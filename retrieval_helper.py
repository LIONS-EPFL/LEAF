import json
import os
import re
import subprocess

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import torch.nn.functional as F
import types

from utils_attacks import encode_text_wrapper_CLIPModel


import torch


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


class CLIPWrapper:
    def __init__(self, model, device, tokenizer, preprocessor):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        

    def encode_text(self, inp, normalize=False):
        encc = types.MethodType(encode_text_wrapper_CLIPModel, self.model)
        text_feats = encc(inp)

        if normalize:
                text_feats = F.normalize(text_feats, dim=-1)
        return text_feats

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        tqdm_loader.set_description("Computing text embeddings")
        for i in tqdm_loader:
            text = texts[i: min(num_text, i + text_batch_size)]
            # text_input = clip.tokenize(text).to(self.device)
            # print(text)
            text_input = self.tokenizer(text).to(self.device)
            text_feats = self.model.get_text_features(text_input)
            if normalize:
                text_feats = F.normalize(text_feats, dim=-1)
            text_embeds.append(text_feats)

        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        image_idx = []
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:
            images = batch["image"]
            
            if "idx" in batch:
                image_idx.extend(batch["idx"])
            #fix this shady hack
            images['pixel_values'] = images['pixel_values'].squeeze(1)
            image_feats = self.model.get_image_features(**images.to(self.device))
            if normalize:
                image_feats = F.normalize(image_feats, dim=-1)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        image_idx = torch.Tensor(image_idx).to(int)
        return image_embeds, image_idx

    @torch.no_grad()
    def get_cosine_similarity_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds, image_idx = self.get_image_embeddings(loader, normalize=True)
        if len(image_idx) != 0:
            text_embeds = text_embeds[image_idx]
        cosine_similarity_scores = self.calc_cosine_similarity(image_embeds, text_embeds)
        return cosine_similarity_scores

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader, pert=False):
        captions = loader.dataset.text if not pert else loader.dataset.pert_text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds, image_idx = self.get_image_embeddings(loader, normalize=True)
        # if len(image_idx) != 0 and args.filter_image_idx:
        #     text_embeds = text_embeds[image_idx]
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores

    def calc_cosine_similarity(self, image_embeds, text_embeds):
        # calculate scores for image-image, text-text, image-text
        cosine_similarity_scores = {}
        # for name, embed1, embed2 in zip(['image-image', 'text-text', 'image-text'], [(image_embeds, image_embeds), (text_embeds, text_embeds), (image_embeds, text_embeds)]):
        for name, embed1, embed2 in [('image-image', image_embeds, image_embeds),
                                     ('text-text', text_embeds, text_embeds),
                                     ('image-text', image_embeds, text_embeds)]:
            # cosine_similarity_scores[name] = {}
            scores = embed1 @ embed2.T
            scores = scores.cpu().numpy()
            for similarity_fn in [np.max, np.min, np.mean]:
                cosine_similarity_scores[f'{name}-{similarity_fn.__name__}'] = similarity_fn(scores)

                if similarity_fn == np.max and name != 'image-text':
                    # calculate the second best score in the case of image-image and text-text in each row
                    second_best_scores = np.partition(scores, -2, axis=1)[:, -2]
                    third_best_scores = np.partition(scores, -3, axis=1)[:, -3]

                    cosine_similarity_scores[f'{name}-{similarity_fn.__name__}'] = similarity_fn(second_best_scores)
                    #
                    # # Mask the diagonal elements
                    # mask = ~np.eye(scores.shape[0], dtype=bool)
                    # masked_scores = scores[mask]
                    # cosine_similarity_scores[f'{name}-{similarity_fn.__name__}'] = similarity_fn(masked_scores)
        return cosine_similarity_scores

    
    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """
        Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.encode_image(i_option.to(self.device)).cpu().numpy()  # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))

            caption_options = []
            for c_option in batch["caption_options"]:
                caption_tokenized = torch.cat([c.unsqueeze(0) if c.dim() == 1 else c for c in [self.tokenizer(c) for c in c_option]])

                # caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
                # caption_tokenized = torch.cat([self.tokenizer(c) for c in c_option])
                caption_embeddings = self.model.encode_text(caption_tokenized.to(self.device)).cpu().numpy()  # B x D
                caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1,
                                                                         keepdims=True)  # B x D
                caption_options.append(np.expand_dims(caption_embeddings, axis=1))

            image_options = np.concatenate(image_options, axis=1)  # B x K x D
            caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L
            scores.append(batch_scores)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class COCO_Retrieval(Dataset):
    def __init__(self, root_dir, image_preprocess=None, max_words=30, split="test",
                 image_perturb_fn=None, download=False, num_samples=-1):
        """
        COCO Retrieval Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: image perturbation function for patch permutation experiments.
        download: Whether to download the dataset if it does not exist.
        """
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory.")

        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test_2017.json'} #updated to 2017
        # filenames = {'val': 'coco_val_karpathy.json', 'test': 'coco_test_karpathy.json'} #TODO changed name
        download_url(urls[split], root_dir)

        # self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
        self.annotation = json.load(open(os.path.join(root_dir, filenames[split])))[:num_samples]
        self.image_preprocess = image_preprocess
        self.image_perturb_fn = image_perturb_fn
        self.image_root = root_dir

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        if self.image_perturb_fn is not None:
            image = self.image_perturb_fn(image)

        return {"image": image, "idx": index}

    def download(self):
        # subprocess.call(["unzip", "train2014.zip"], cwd=self.root_dir)

        # subprocess.call(["wget", "http://images.cocodataset.org/zips/val2014.zip"], cwd=self.root_dir)
        # subprocess.call(["unzip", "val2014.zip"], cwd=self.root_dir)

        subprocess.call(["wget", "http://images.cocodataset.org/zips/test2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T  # Make it N_ims x N_text

        else:
            scores_t2i = scores
            scores_i2t = scores

        print(f"COCO results across {scores_i2t.shape} samples. ")
        prec_at_1 = AverageMeter()
        prec_at_5 = AverageMeter()

        # Text retrieval
        tqdm_iterator = tqdm(range(len(self.img2txt)))
        for i in tqdm_iterator:
            top5_captions = np.argsort(scores_i2t[i])[-5:]
            true_captions = self.img2txt[i]

            prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:])) > 0)
            prec_at_5.update(len(set(true_captions) & set(top5_captions)) > 0)

            tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f}")

        # Image Retrieval
        image_prec_at_1 = AverageMeter()
        image_prec_at_5 = AverageMeter()

        tqdm_iterator = tqdm(range(len(self.txt2img)))
        for i in tqdm_iterator:
            top5_images = np.argsort(scores_t2i[:, i])[-5:]
            true_image = self.txt2img[i]
            image_prec_at_1.update(true_image in top5_images[-1:])
            image_prec_at_5.update(true_image in top5_images)

            tqdm_iterator.set_description(
                f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f}")

        records = {"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg, "TextPrec@1": prec_at_1.avg,
                    "TextPrec@5": prec_at_5.avg}
        return records
