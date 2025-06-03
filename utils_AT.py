import torch
import string

import json
import logging
import math
import os
import time
from PIL import Image

import numpy as np
import pandas as pd
import torch.nn.functional as F
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.distributed import is_master
from open_clip_train.precision import get_autocast

from tqdm import tqdm
from copy import deepcopy

from utils_attacks import attack_image_classification, attack_text_charmer_classification, attack_image, attack_text, attack_text_charmer_inference, convert_clip_text_model


def get_vocabulary(dataset, dataset_name):
    '''
    get the characted volabulary from a dataset
    '''
    V = set([-1]) # Remove character operator
    if dataset_name in ['mnli', 'rte', 'qnli']:
        keyword = 'hypothesis'
    elif dataset_name in ['imdb','yelp','agnews', 'rotten_tomatoes']:
        keyword = 'text'
    else:
        keyword = 'sentence'
    for x in dataset:
        V = V.union([ord(y) for y in set(x[keyword])])
    return list(V)

class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, normalize, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    
    top1, top5, n, top1_adv = 0., 0., 0., 0.
    for i,(images, target) in tqdm(enumerate(dataloader), unit_scale=args.batch_size):
        # #we eval only on 10 batches
        # if i==10: break
        images = images.to(device=args.device, dtype=input_dtype)
        target = target.to(args.device)

        with autocast():
            # predict
            with torch.no_grad():
                output = model(image=normalize(images))
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # attack and predict
        adv_images = attack_image_classification(model, normalize, images.detach(), classifier.detach(), target, args.device, eps=args.eps_adv, n_steps=args.n_steps_adv, stepsize=args.stepsize_adv, debug=False)
        with autocast():
            with torch.no_grad():
                output_adv = model(image=normalize(adv_images))
                image_features_adv = output_adv['image_features'] if isinstance(output, dict) else output[0]
                logits_adv = 100. * image_features_adv @ classifier

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc1_adv = accuracy(logits_adv, target, topk=(1,))[0]
        top1 += acc1
        top1_adv += acc1_adv
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n)
    top1_adv = (top1_adv / n)
    top5 = (top5 / n)
    return top1, top5, top1_adv

def run_text_classification(model, image_features, dataset, V, template, args, tokenizer):
    '''
    does zero shot text classification using the model and image features
    '''

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    acc, acc_adv, n = 0., 0., 0.
    for i,d in enumerate(tqdm(dataset)):
        # we evaluate on 100 examples only
        if i==args.n_val_text: break
        sentence, label = d['text'], d['label']

        perturbed_sentence, dist = attack_text_charmer_classification(model,tokenizer,sentence,image_features,label,args.device,n=args.n_charmer_test,k=args.k_adv_test,V=V,debug=False)

        tokens = tokenizer([template.format(sentence), template.format(perturbed_sentence)]).to(args.device)
        text_features = model.encode_text(tokens).view(2,-1)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (text_features @ image_features.transpose(-1,-2)).softmax(dim=-1)
        
        n+=1
        acc+=(label==torch.argmax(text_probs,dim=-1)[0].item())
        acc_adv+=(label==torch.argmax(text_probs,dim=-1)[1].item())
    return acc/n, acc_adv/n


def zero_shot_eval(model, preprocess_without_normalize, normalize, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'val-text-classification' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5, top1_adv = run(model, classifier, normalize, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top1-adv'] = top1_adv
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5, top1_adv = run(model, classifier, normalize, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top1-adv'] = top1_adv
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    logging.info('Starting zero-shot text classification.')
    if 'val-agnews' in data:
        
        data_dict = data['val-agnews']

        images = [Image.open(img_path) for img_path in data_dict['img_list']]
        with torch.no_grad():
            images = torch.cat([preprocess_without_normalize(img).unsqueeze(0) for img in images],dim=0).to(args.device)
            image_features = model.encode_image(normalize(images)).view(len(images),-1)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            acc, acc_adv = run_text_classification(model,image_features, data_dict['test_set'],data_dict['V'], data_dict['template'], args, tokenizer)
            results['agnews-zeroshot-val-acc'] = acc
            results['agnews-zeroshot-val-acc-adv'] = acc_adv

    if 'val-sst2' in data:
        
        data_dict = data['val-sst2']

        images = [Image.open(img_path) for img_path in data_dict['img_list']]
        with torch.no_grad():
            images = torch.cat([preprocess_without_normalize(img).unsqueeze(0) for img in images],dim=0).to(args.device)
            image_features = model.encode_image(normalize(images)).view(len(images),-1)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            acc, acc_adv = run_text_classification(model,image_features, data_dict['test_set'],data_dict['V'], data_dict['template'], args, tokenizer)
            results['sst2-zeroshot-val-acc'] = acc
            results['sst2-zeroshot-val-acc-adv'] = acc_adv

    if 'train-agnews' in data:
        
        data_dict = data['train-agnews']

        images = [Image.open(img_path) for img_path in data_dict['img_list']]
        with torch.no_grad():
            images = torch.cat([preprocess_without_normalize(img).unsqueeze(0) for img in images],dim=0).to(args.device)
            image_features = model.encode_image(normalize(images)).view(len(images),-1)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            acc, acc_adv = run_text_classification(model,image_features, data_dict['test_set'],data_dict['V'], data_dict['template'], args, tokenizer)
            results['agnews-zeroshot-train-acc'] = acc
            results['agnews-zeroshot-train-acc-adv'] = acc_adv

    if 'train-sst2' in data:
        
        data_dict = data['train-sst2']

        images = [Image.open(img_path) for img_path in data_dict['img_list']]
        with torch.no_grad():
            images = torch.cat([preprocess_without_normalize(img).unsqueeze(0) for img in images],dim=0).to(args.device)
            image_features = model.encode_image(normalize(images)).view(len(images),-1)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            acc, acc_adv = run_text_classification(model,image_features, data_dict['test_set'],data_dict['V'], data_dict['template'], args, tokenizer)
            results['sst2-zeroshot-train-acc'] = acc
            results['sst2-zeroshot-train-acc-adv'] = acc_adv
    
    #del images, image_features, classifier, data_dict

    return results

def train_one_epoch_text_only(model, model_frozen, tokenizer, V, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}

    times = []

    losses_accum = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        _, texts = batch
        model.eval()
        '''
        FARE objective only
        '''
        with torch.no_grad():
            text_features_frozen = model_frozen.encode_text(tokenizer(texts).to(device=device, non_blocking=True), normalize=args.normalize_fare)
            if args.use_charmer:
                start_time = time.time()
                adv_texts = []
                for j,t in enumerate(texts):
                    adv_text,_ = attack_text_charmer_inference(model,tokenizer,t,text_features_frozen[j],device,objective='l2',n=args.rho,k=args.k_adv,constrain=args.constrain,V=V,debug=False)
                    adv_texts.append(adv_text)
                end_time = time.time()
                times.append(end_time - start_time)
            else:
                start_time = time.time()
                _, adv_texts = attack_text(model,tokenizer,texts,text_features_frozen,device,objective='l2',n=args.rho,k=args.k_adv,V=V,constrain=args.constrain,debug=False)        
                end_time = time.time()
                times.append(end_time - start_time)

        pd.DataFrame(times).to_csv(f'times_{args.use_charmer}.csv', index=False)
        adv_texts = tokenizer(adv_texts).to(device=device, non_blocking=True)

        '''
        FARE things:
        '''
        with autocast():    
            model.train()
            text_features_adv = model.encode_text(adv_texts, normalize=args.normalize_fare)
        
            loss_FARE_text = F.mse_loss(text_features_frozen,
                                        text_features_adv,reduction='none').sum(dim=-1).mean()

        data_time_m.update(time.time() - end)

        '''
        TOTAL loss:
        '''
        total_loss = loss_FARE_text / args.accum_freq
        if 'loss_FARE_text' not in losses_accum:
            losses_accum["loss"] = total_loss
            losses_accum['loss_FARE_text'] = loss_FARE_text / args.accum_freq
        else:
            losses_accum["loss"] += total_loss 
            losses_accum['loss_FARE_text'] += loss_FARE_text / args.accum_freq
        
        backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize() and (i+1)%args.accum_freq == 0:
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                if (i+1)%args.accum_freq == 0:
                    scaler.step(optimizer)
            if (i+1)%args.accum_freq == 0:
                scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            if (i+1)%args.accum_freq == 0:
                optimizer.step()
        
        if (i+1)%args.accum_freq == 0:
            optimizer.zero_grad()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i+1)%args.accum_freq == 0 and (batch_count % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(texts)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses_accum.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for
        if (i+1)%args.accum_freq == 0:
            losses_accum = {}
    return log_data

def evaluate(model, model_frozen, preprocess_without_normalize, normalize, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, preprocess_without_normalize, normalize, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                if tokenizer is not None:
                    texts = tokenizer(texts).to(device=device, non_blocking=True)
                else:
                    texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(normalize(images), texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics, log_data

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
    
if __name__ == "__main__":
    pass