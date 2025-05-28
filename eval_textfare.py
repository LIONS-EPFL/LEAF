import os
from tqdm import tqdm
import types
import torch
from PIL import Image
import argparse
import pandas as pd
from data_AT import get_text_classification_dataset
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPTextConfig
from utils_attacks import encode_text_wrapper, encode_text_wrapper_CLIPModel, tokenizer_wrapper, attack_text_charmer_inference, attack_text_charmer, attack_text_bruteforce

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--attack_name", 
        type=str, 
        default="leaf"
        )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2"
        )
    parser.add_argument(
        "--k",
        type=int,
        default=1
        )
    parser.add_argument(
        "--n_charmer",
        type=int,
        default=20
        )
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="number of sentences to compute the clean and adversarial accuracy"
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1280,
        help="number of sentences to compute the clean and adversarial accuracy"
        )
    parser.add_argument(
        "--constrain",
        action="store_true",
        default=False,
        help="Constrain the adversarial attacks to not produce words with meaning",
        )
    args = parser.parse_args()
    
    # load processor
    if "ViT-L" in args.model_name or "vit-large" in args.model_name:
        processor_name = 'openai/clip-vit-large-patch14'
        clean_model_name = 'openai/clip-vit-large-patch14'
    elif "ViT-H" in args.model_name:
        processor_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        clean_model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    elif "ViT-g" in args.model_name:
        processor_name = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
        clean_model_name = 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K'
    else:
        #bigG
        processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    clip_processor = AutoProcessor.from_pretrained(processor_name)

    # load data
    if args.dataset == "agnews":
        data = get_text_classification_dataset('fancyzhx/ag_news',n_samples=args.n_test,test=True)
    elif args.dataset == "sst2":
        data = get_text_classification_dataset('stanfordnlp/sst2',n_samples=args.n_test,test=True)
    elif args.dataset == "imdb":
        data = get_text_classification_dataset('stanfordnlp/imdb',n_samples=args.n_test,test=True)
    elif args.dataset == "yelp":
        data = get_text_classification_dataset('fancyzhx/yelp_polarity',n_samples=args.n_test,test=True)

    #dataset = data['val-' + args.dataset]

    dataset, V, template = data['test_set'],data['V'], data['template']

    # load clean model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean_model = CLIPModel.from_pretrained(clean_model_name).to(device)
    clean_model.eval()
    
    # wrap it
    clean_model.encode_text = types.MethodType(encode_text_wrapper_CLIPModel, clean_model)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # wrap it
    model.encode_text = types.MethodType(encode_text_wrapper_CLIPModel, model)
    tokenizer = tokenizer_wrapper(clip_processor.tokenizer)

    # filename
    os.makedirs('results_textfare', exist_ok=True)
    if len(args.model_name.split('/')) == 2:
        filename = 'results_textfare/' + args.model_name.split('/')[-1] + '_' + args.dataset + f'_' + args.attack_name + f'_k{args.k}_n_charmer_{args.n_charmer}' + ('_constrained' if args.constrain else '') + '.csv'
    else:
        filename = 'results_textfare/' + args.model_name.split('/')[-2] + '_' + args.dataset + f'_' + args.attack_name + f'_k{args.k}_n_charmer_{args.n_charmer}' + ('_constrained' if args.constrain else '') + '.csv'

    results = {'sentence':[], 'adv_sentence':[], 'textfare_clean':[], 'textfare_adv':[]}
    acc, acc_adv, n = 0., 0., 0.
    with torch.no_grad():
        for i,d in enumerate(tqdm(dataset)):
            # we evaluate on 100 examples only
            if i==args.n_test: break
            sentence, label = d['text'], d['label']

            tokens_original = tokenizer([sentence]).to(device)
            
            original_clean_features = clean_model.encode_text(tokens_original,normalize=False)
            original_features = model.encode_text(tokens_original,normalize=False)


            if args.attack_name == 'leaf':
                _, perturbed_sentence = attack_text_charmer(model,tokenizer,[sentence],original_features,device,objective='l2',n=args.n_charmer,k=args.k,V=V,debug=False, constrain=args.constrain)
                perturbed_sentence = perturbed_sentence[0]
            elif args.attack_name == 'charmer':
                perturbed_sentence, dist = attack_text_charmer_inference(model,tokenizer,sentence,original_features,device,objective='l2',n=args.n_charmer,k=args.k,V=V,debug=False,batch_size=args.batch_size, constrain=args.constrain)
            elif args.attack_name == 'bruteforce':
                perturbed_sentence, dist = attack_text_bruteforce(model,tokenizer,sentence,original_features,device,objective='l2',V=V,debug=False,batch_size=128*10, constrain=args.constrain)

            tokens = tokenizer([perturbed_sentence]).to(device)

            text_features = model.encode_text(tokens,normalize=False)

            #print(text_features.shape, original_features.shape)

            loss_clean = ((original_clean_features-original_features)**2).sum().item()
            loss_adv = ((original_clean_features-text_features)**2).sum().item()
            
            n+=1
            results['sentence'].append(sentence)
            results['adv_sentence'].append(perturbed_sentence)
            results['textfare_clean'].append(loss_clean)
            results['textfare_adv'].append(loss_adv)
            pd.DataFrame(results).to_csv(filename,index=False)
            print(args.attack_name,loss_clean, loss_adv)
        
