import os
from tqdm import tqdm
import types
import torch
from PIL import Image
import argparse
import pandas as pd
from data_AT import get_text_classification_dataset
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPTextConfig
from utils_attacks import encode_text_wrapper, encode_text_wrapper_CLIPModel, tokenizer_wrapper, attack_text_charmer_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None
        )
    parser.add_argument(
        "--label_encoder", 
        type=str, 
        default="image"
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
        default=1000,
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
    if "ViT-L" in args.model_name:
        processor_name = 'openai/clip-vit-large-patch14'
    elif "ViT-H" in args.model_name:
        processor_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    elif "ViT-g" in args.model_name:
        processor_name = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
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

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()
    # wrap it
    model.encode_text = types.MethodType(encode_text_wrapper_CLIPModel, model)
    tokenizer = tokenizer_wrapper(clip_processor.tokenizer)

    if args.label_encoder == 'image':
        images = [Image.open(img_path) for img_path in data['img_list']]
        with torch.no_grad():
            images_processed = clip_processor(images=images, return_tensors="pt").to(device)
            label_features = model.get_image_features(**images_processed)
            label_features /=label_features.norm(dim=-1, keepdim=True)
    elif args.label_encoder == 'text':
        labels = data['caption_list']
        with torch.no_grad():
            tokens_labels = tokenizer(labels).to(device)
            label_features = model.encode_text(tokens_labels,normalize=True)

    # filename
    os.makedirs('results_zero_shot_text', exist_ok=True)
    if len(args.model_name.split('/')) == 2:
        filename = 'results_zero_shot_text/' + args.model_name.split('/')[-1] + '_' + args.dataset + f'_k{args.k}_n_charmer_{args.n_charmer}' + ('_constrained' if args.constrain else '') + ('_text_only' if args.label_encoder == 'text' else '') + '.csv'
    else:
        filename = 'results_zero_shot_text/' + args.model_name.split('/')[-2] + '_' + args.dataset + f'_k{args.k}_n_charmer_{args.n_charmer}' + ('_constrained' if args.constrain else '') + ('_text_only' if args.label_encoder == 'text' else '') + '.csv'

    results = {'sentence':[], 'original_label':[], 'predicted_label':[], 'adv_sentence':[], 'adv_label':[]}
    acc, acc_adv, n = 0., 0., 0.
    with torch.no_grad():
        for i,d in enumerate(tqdm(dataset)):
            # we evaluate on 100 examples only
            if i==args.n_test: break
            sentence, label = d['text'], d['label']

            perturbed_sentence, dist = attack_text_charmer_classification(model,tokenizer,sentence,label_features,label,device,n=args.n_charmer,k=args.k,V=V,debug=False,batch_size=128*20)

            #tokens = tokenizer([template.format(sentence), template.format(perturbed_sentence)]).to(device)
            tokens = tokenizer([sentence, perturbed_sentence]).to(device)

            text_features = model.encode_text(tokens).view(2,-1)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (text_features @label_features.transpose(-1,-2)).softmax(dim=-1)
            
            n+=1
            acc+=(label==torch.argmax(text_probs,dim=-1)[0].item())
            acc_adv+=(label==torch.argmax(text_probs,dim=-1)[1].item())
            results['sentence'].append(sentence)
            results['original_label'].append(label)
            results['predicted_label'].append(torch.argmax(text_probs,dim=-1)[0].item())
            results['adv_sentence'].append(perturbed_sentence)
            results['adv_label'].append(torch.argmax(text_probs,dim=-1)[1].item())
            pd.DataFrame(results).to_csv(filename,index=False)
            print(acc,acc_adv)
        
