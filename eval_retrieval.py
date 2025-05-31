import argparse  
import os
import pathlib
import time
from pathlib import Path

import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from retrieval_helper import *
from utils_attack import attack_text_charmer_constrained_ret

COCO2017_DIR = pathlib.Path("PATH-TO-COCO")

retrievalTargets = ["A man aggressively kicks a stray dog on the street.", "This is an image of a a pyramid", "A group of teenagers vandalizes a public statue.",]

#preprocessor-name, LEAF-model
modelDict = {'ViT-L-14': ["openai/clip-vit-large-patch14", "LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2"],
        'ViT-H-14': ["laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "LEAF-CLIP/CLIP-ViT-H-rho50-k1-constrained-FARE2"],
        'ViT-bigG-14': ["laion/CLIP-ViT-g-14-laion2B-s12B-b42K", "LEAF-CLIP/CLIP-ViT-g-rho50-k1-constrained-FARE2"],
        'ViT-bigG-14': ["laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "LEAF-CLIP/CLIP-ViT-bigG-rho50-k1-constrained-FARE2"]
         }


class tokenizer_wrapper():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, x):
        return torch.tensor(self.tokenizer(x,padding=True,truncation=True).input_ids)

def zero_shot_retrival(model, model_name, preprocess_val, tokenizer, k=1 ,n=10, num_samples=1000, obj='l2', target=None,  device='cuda'):

    model = CLIPWrapper(model, device, tokenizer, preprocess_val)

    out_folder = './retrieval_evals'
    os.makedirs(out_folder,exist_ok=True)

    data_dir = Path(COCO2017_DIR)
    
    max_words = 50  
    split = "test"
    
    dataset = COCO_Retrieval(root_dir=data_dir, split=split, image_preprocess=preprocess_val, image_perturb_fn=None,
                             max_words=max_words, download=False, num_samples=num_samples)
    collate_fn = _default_collate if preprocess_val is None else None
    loader = DataLoader(dataset, batch_size=25, shuffle=False, num_workers=8, collate_fn=collate_fn)

    #clean evaluation
    scores = model.get_retrieval_scores_dataset(loader)
    result_records = dataset.evaluate_scores(scores)


    target_cap = retrievalTargets[target]
    anchor_feat = tokenizer([target_cap]).to(device)
    anchor_features = model.model.get_text_features(**anchor_feat)
    print("Running attack with target: ", target_cap)
    
    ## Output file that stores the original sentences and perturbed ones
    out_file_p = f'perturbations_{model_name}_' + 'coco2017' + f'_samples_{num_samples}_{target}_{obj}'+ f'_k{k}_n{n}' + '.csv'
    ## Output file to store the retrieval results
    out_file_r = f'results_{model_name}_' + 'coco2017' + f'_samples_{num_samples}_targ_{target}_{obj}'+ f'_k{k}_n{n}' + '.json'

    clean_sentences = loader.dataset.text

    pert = []
    dists = []
    clean_sents = []
    times = []
    df = {}

    with torch.no_grad():
        for i, sent in enumerate(tqdm(clean_sentences)):
            
            start = time.time()
            perturbed_sentence, dist = attack_text_charmer_constrained_ret(model, tokenizer, sent, anchor_features, device, objective=obj, n=n,k=k, debug=False)
            pert.append(perturbed_sentence)
            clean_sents.append(sent)
            dists.append(dist)
            end = time.time()
            times.append(end-start)
        df['sentence'] = clean_sents
        df['sentence_adv'] = pert
        df['distance'] = dists
        df['time'] = times
        pd.DataFrame(df).to_csv(os.path.join(out_folder, out_file_p),index=False)

    loader.dataset.pert_text =  df['sentence_adv']
    adv_scores = model.get_retrieval_scores_dataset(loader, pert=True)
    result_records_adv = dataset.evaluate_scores(adv_scores)

    outs = {'clean': result_records, 'adv': result_records_adv}

    json.dump(outs, open(os.path.join(out_folder, out_file_r), 'w'), indent=4)
    print("Model: {} \n Target: {} results: {}".format(pretraining_name, model_name, target, outs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CHARMER constrained retrieval evaluations", add_help=False
    )
    parser.add_argument(
        "--n", default=10, type=int, help="parameter n in charmer"
    )
    parser.add_argument(
        "--k", default=2, type=int, help="parameter k in charmer"
    )
    parser.add_argument(
        "--num-samples", default=100, type=int, help="num samples to evaluate for"
    )
    parser.add_argument(
        "--obj", default='dissim', type=str, help="l2/dissim"
    )
    parser.add_argument(
        "--model-name", default='ViT-L-14', type=str, choices=["ViT-L-14", "ViT-H-14", "ViT-bigG-14"]
    )
    parser.add_argument(
        "--target", default=0, type=int, help="Target caption index?(0,1,2)"
    )

    args = parser.parse_args()    

    model = CLIPModel.from_pretrained(modelDict[args.model_name][1])

    processor_name = modelDict[args.model_name][0]     
    clip_processor = AutoProcessor.from_pretrained(processor_name)
    tokenizer = lambda x: clip_processor(text=x, padding=True, return_tensors='pt')
    preprocess = lambda x: clip_processor(images=x, return_tensors="pt")
        
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    zero_shot_retrival(model, args.model_name, preprocess, tokenizer, args.k, args.n, args.num_samples, args.obj, args.target, device)
    
