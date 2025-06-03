import json
import logging
import sys
import os
import time
import string
import random
import argparse

import numpy as np
import transformers

import open_clip
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel, AutoModel
from torchvision import transforms
from tqdm import tqdm

from open_clip import OPENAI_IMAGENET_TEMPLATES
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
from open_clip.zero_shot_classifier import build_zero_shot_classifier
from robust_vlm.train.utils import cosine_lr
from robust_vlm.train.pgd_train import pgd
from robust_vlm.train.apgd_train import apgd_train as apgd
import wandb
from robust_vlm.train.utils import init_wandb, AverageMeter, str2bool, unwrap_model
from robust_vlm.train.datasets import ImageNetDataset

from robust_vlm.train.log import setup_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--template', type=str, default='ensemble')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether the embedding is normalized')
parser.add_argument('--start_step', type=int, default=0, help='Start step for training (for continuing runs)')
parser.add_argument('--optimizer_state', type=str, default='', help='Path to optimizer state file for continuing runs')
parser.add_argument('--steps', type=int, default=20000, help='Number of training steps')
parser.add_argument('--warmup', type=int, default=14000, help='Warmup steps')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--loss', type=str, default='l2', help='ce, l2')
parser.add_argument('--loss_clean', type=str, default='none', help='ce, l2')
parser.add_argument('--clean_weight', type=float, default=0., help='Weight for clean loss')
parser.add_argument('--trades', type=str2bool, default=False, help='Use TRADES')
parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type; sgd, adamw')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--attack', type=str, default='pgd', help='Adversarial attack type')
parser.add_argument('--inner_loss', type=str, default='l2', help='Inner loss function for adversarial training')
parser.add_argument('--norm', type=str, default='linf', help='Norm for adversarial perturbation')
parser.add_argument('--eps', type=float, default=4, help='Epsilon for adversarial perturbation')
parser.add_argument('--iterations_adv', type=int, default=10, help='Iterations for adversarial attack')
parser.add_argument('--stepsize_adv', type=float, default=1., help='Step size for adversarial attack (no effect for apgd)')
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=200, help='Evaluation frequency')
parser.add_argument('--output_dir', type=str, default='', help='Output directory')
parser.add_argument('--save_checkpoints', type=str2bool, default=False, help='Save 10 training checkpoints')
parser.add_argument('--skip-first-val', type=str2bool, default=False, help='Skip first validation')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')


def main(args):
    # either resume or start a fresh wandb run
    if args.wandb:
        # If continuing, resume wandb. If fresh, start anew
        if args.start_step > 0:
            # "resume='allow'" will pick up a previous run in the same folder if it exists.
            # You can also pass a known run ID with `id='abc123', resume='must'`.
            init_wandb(
                project_name='clip-finetune',
                model_name=args.finetuned_model_name,
                config=vars(args),
                id=args.finetuned_model_name.replace("_temp", "").split("_")[-1],
                resume='allow',
            )
        else:
            init_wandb(
                project_name='clip-finetune',
                model_name=args.finetuned_model_name,  # or a custom name
                config=vars(args),
                id=random_str,
            )
    else:
        wandb.init(mode='disabled')

    logging.info("Starting script ...")

    # Write arguments to logging
    logging.info(f"Arguments:")
    logging.info("\n".join(f"{arg}: {value}" for arg, value in vars(args).items()))


    main_device = 0
    if args.optimizer_state != '':
        # Make sure we are continuing at the correct step, etc.
        assert args.start_step > 0
        assert str(args.start_step) in args.optimizer_state
        # We'll allow user to specify the pretrained=none or actual checkpoint, etc.
        if args.pretrained in ['', 'none']:
            args.pretrained = args.optimizer_state.replace('_opt', '')

    if args.model_name.startswith('hf-hub:'):
        model_orig, _, image_processor = open_clip.create_model_and_transforms(args.model_name)
        model, _, _ = open_clip.create_model_and_transforms(args.model_name)
        tokenizer = open_clip.get_tokenizer(args.model_name)
    else:
        # ...
        assert args.pretrained in ['', 'none']
        raise NotImplementedError
        model_orig = AutoModel.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        image_processor, tokenizer = get_preprocessor(args.model_name)

    # Remove the Normalize transform by creating a new Compose object
    preprocessor_without_normalize = transforms.Compose(image_processor.transforms[:-1])
    normalize = image_processor.transforms[-1]
    del image_processor
    logging.info(f'[preprocessor_without_normalize] {preprocessor_without_normalize}')
    logging.info(f'[normalize] {normalize}')

    # get data
    if args.dataset == 'imagenet':
        dataset = ImageNetDataset(
            root=args.imagenet_root + '/train',
            transform=preprocessor_without_normalize,
        )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=8, drop_last=True)

    dataset_eval = ImageNetDataset(
        root=args.imagenet_root + '/val',
        transform=preprocessor_without_normalize,
    )
    generator = np.random.default_rng(seed=0)
    rand_idcs = generator.choice(len(dataset_eval), 1000, replace=False)
    dataset_eval = torch.utils.data.Subset(dataset_eval, rand_idcs)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, drop_last=False)

    # zero-shot classifier
    model_orig.to(main_device)
    if args.template == 'ensemble':
        templates = OPENAI_IMAGENET_TEMPLATES
    elif args.template == 'std':
        templates = [lambda c: f'This is a photo of a {c}']
    else:
        raise ValueError(f'Unknown template: {args.template}')

    model_name_cleaned = args.model_name.replace("/", "-").replace(":", "-")
    if os.path.exists(p:=f'{args.output_dir}/embedding_text_labels_norm.pt'):
        embedding_text_labels_norm = torch.load(p)
    elif os.path.exists(p:=f'/tmp/{model_name_cleaned}_embedding_text_labels_norm.pt'):
        embedding_text_labels_norm = torch.load(p)
    else:
        embedding_text_labels_norm = build_zero_shot_classifier(
            model=model_orig,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=50,
            device=f'cuda:{main_device}',
            use_tqdm=True
        )
        # save the embedding_text_labels_norm
        torch.save(embedding_text_labels_norm, f'{args.output_dir}/embedding_text_labels_norm.pt')
        # also save to tmp
        torch.save(embedding_text_labels_norm, f'/tmp/{model_name_cleaned}_embedding_text_labels_norm.pt')

    # Wrap the model
    model_orig.cpu()
    model_orig = ClipVisionModel(model=model_orig, args=args, normalize=normalize)
    model = ClipVisionModel(model=model, args=args, normalize=normalize)

    if args.pretrained not in ['', 'none']:
        logging.info(f'Loading pretrained model from {args.pretrained}')
        model.model.load_state_dict(torch.load(args.pretrained))

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model_orig = torch.nn.DataParallel(model_orig)
        model = torch.nn.DataParallel(model)
    model_orig.cuda()
    model.cuda()

    # Possibly save checkpoint 0
    if args.save_checkpoints and args.start_step == 0:
        torch.save(unwrap_model(model).model.state_dict(),
                   f'{args.output_dir}/checkpoints/step_0.pt')

    # Evaluate before training if new run
    if args.start_step == 0 and not args.skip_first_val:
        logging.info('Evaluating model before training')
        eval_logs = evaluate(model, dataloader_eval, embedding_text_labels_norm, args)
        wandb.log(eval_logs)

    # set up optimizer
    params = unwrap_model(model).model.parameters()
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.wd)
    else:
        raise ValueError(f'Optimizer {args.opt} not supported.')

    if args.optimizer_state != '':
        optimizer.load_state_dict(torch.load(args.optimizer_state, weights_only=False))

    # set scheduler
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)
    scheduler(args.start_step)

    # compute amount of epochs
    total_epochs = args.steps / len(dataloader)
    logging.info(f'train for {total_epochs} epochs')
    args.total_epochs = total_epochs

    # train
    step_total = args.start_step
    epoch = 0
    while step_total < args.steps:
        step_total = train_one_epoch(
            step_total,
            model=model,
            model_orig=model_orig,
            dataloader=dataloader,
            dataloader_eval=dataloader_eval,
            optimizer=optimizer,
            scheduler=scheduler,
            embedding_text_labels_norm=embedding_text_labels_norm,
            normalize=normalize,
            args=args,
            epoch=epoch
        )
        logging.info(f'Epoch {epoch} done.')
        epoch += 1

    # save final model
    torch.save(unwrap_model(model).model.state_dict(),
               f'{args.output_dir}/checkpoints/final.pt')
    torch.save(optimizer.state_dict(),
               f'{args.output_dir}/checkpoints/final_opt.pt')

    # remove old fallback files
    for file in os.listdir(f'{args.output_dir}/checkpoints'):
        if file.startswith('fallback'):
            os.remove(f'{args.output_dir}/checkpoints/{file}')

    # rename temp dir if needed
    if args.output_dir.endswith('_temp'):
        os.rename(args.output_dir, args.output_dir[:-5])


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, args, normalize):
        super().__init__()
        self.model = model
        self.args = args
        self.normalize = normalize
        self.is_hf_clip = isinstance(model, transformers.CLIPModel)
        if not self.is_hf_clip:
            # If open_clip or something else, just keep the .visual part
            self.model = self.model.visual

    def forward(self, image, output_normalize):
        if self.is_hf_clip:
            embedding = self.model.get_image_features(pixel_values=self.normalize(image))
        else:
            embedding = self.model(self.normalize(image))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


class ComputeLossWrapper:
    def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean',
                 loss=None, logit_scale=100., targeted=False):
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale
        self.targeted = targeted

    def __call__(self, embedding, targets, **kwargs):
        loss = compute_loss(
            loss_str=self.loss_str,
            embedding=embedding,
            targets=targets,
            embedding_orig=self.embedding_orig,
            logit_scale=self.logit_scale,
            embedding_text_labels_norm=self.embedding_text_labels_norm,
            reduction=self.reduction,
        )
        if self.targeted:
            loss *= -1
        return loss


def train_one_epoch(
    step_total, model, model_orig, dataloader, optimizer, scheduler, normalize,
    embedding_text_labels_norm, args, epoch, dataloader_eval=None
):
    model_orig.eval()
    model.train()

    loss_meter = AverageMeter('loss')
    cos_sim_meter = AverageMeter('cos-sim')
    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')

    epoch_start_time = time.time()
    for i, (data, targets) in enumerate(dataloader):
        is_classification = isinstance(targets, torch.Tensor)
        data = data.cuda()
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.cuda()

        with torch.no_grad():
            embedding_orig = model_orig(image=data, output_normalize=args.output_normalize)

        # loss for the attack
        loss_inner_wrapper = ComputeLossWrapper(
            embedding_orig, embedding_text_labels_norm,
            reduction='none' if args.attack == 'apgd' else 'mean',
            loss=args.inner_loss,
            logit_scale=100.0,
        )
        model.eval()

        if args.attack == 'pgd':
            data_adv = pgd(
                forward=model,
                loss_fn=loss_inner_wrapper,
                data_clean=data,
                targets=targets,
                norm=args.norm,
                eps=args.eps,
                iterations=args.iterations_adv,
                stepsize=args.stepsize_adv,
                output_normalize=args.output_normalize,
                perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                mode='max',
                verbose=False
            )
        elif args.attack == 'apgd':
            # apgd currently always applies output normalization
            data_adv = apgd(
                model=model,
                loss_fn=loss_inner_wrapper,
                x=data,
                y=targets,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.iterations_adv,
                verbose=True
            )
        elif args.attack == 'none':
            data_adv = data

        del loss_inner_wrapper
        model.train()

        embedding_clean = model(data, output_normalize=args.output_normalize)
        if args.clean_weight > 0.:
            loss_clean = compute_loss(
                loss_str=args.loss_clean,
                embedding=embedding_clean,
                targets=targets,
                embedding_orig=embedding_orig,
                logit_scale=100.,
                embedding_text_labels_norm=None
            )
        else:
            loss_clean = 0.

        embedding_adv = model(data_adv, output_normalize=args.output_normalize)
        del data, data_adv

        if args.trades:
            embedding_clean_no_grad = embedding_clean.detach().clone()
            embedding_orig.cpu()

        loss = compute_loss(
            loss_str=args.loss,
            embedding=embedding_adv,
            targets=targets,
            embedding_orig=embedding_orig if not args.trades else embedding_clean_no_grad,
            logit_scale=100.,
            embedding_text_labels_norm=embedding_text_labels_norm
        )
        loss_total = args.clean_weight * loss_clean + (1 - args.clean_weight) * loss
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_total += 1
        scheduler(step_total)

        with torch.no_grad():
            # only for logging
            embedding_orig.cuda()
            cos_sim_clean = F.cosine_similarity(embedding_clean, embedding_orig, dim=1).mean()
            cos_sim = F.cosine_similarity(embedding_adv, embedding_orig, dim=1).mean()
            if is_classification:
                logits_adv = embedding_adv @ embedding_text_labels_norm
                racc = compute_acc(logits_adv, targets)
                embedding_clean_norm = F.normalize(embedding_clean, dim=1)
                logits_clean = embedding_clean_norm @ embedding_text_labels_norm
                acc = compute_acc(logits_clean, targets)
                acc_meter.update(acc, n_samples)
                racc_meter.update(racc, n_samples)
                del embedding_clean_norm, embedding_clean
            else:
                acc = None
                racc = None

        loss_meter.update(loss.item(), n_samples)
        cos_sim_meter.update(cos_sim.item(), n_samples)

        eval_logs = {}
        if step_total % args.eval_freq == 0:
            logging.info(f'Running evaluation at step {step_total}')
            eval_logs = evaluate(model, dataloader_eval, embedding_text_labels_norm, args)
            model.train()

        lr_ = optimizer.param_groups[0].get('lr')
        if step_total % args.log_freq == 0:
            # compute expected average epoch time in hours
            batch_avg_time = (time.time() - epoch_start_time) / (i + 1) / 3600.0
            epoch_avg_time = batch_avg_time * len(dataloader)
            this_epoch_remaining = epoch_avg_time - (time.time() - epoch_start_time) / 3600.0
            total_remaining = epoch_avg_time * (args.total_epochs - epoch - i/len(dataloader))

            log_str = (f'[step] {step_total} [lr] {lr_:.6f} [loss] {loss.item():.6f} '
                       f'[cos-sim] {cos_sim.item():.3f}')
            if is_classification:
                log_str += f' [acc] {acc:.2f} [racc] {racc:.2f}'
            log_str += (f' [epoch avg time] {epoch_avg_time:.2f}h '
                        f'[this epoch remain] {this_epoch_remaining:.2f}h '
                        f'[total remain] {total_remaining:.2f}h')

            logging.info(log_str)
            log_data = {
                'step': step_total,
                'lr': lr_,
                'loss': loss.item(),
                'loss-total': loss_total.item(),
                'cos-sim-clean': cos_sim_clean.item(),
                'cos-sim': cos_sim.item(),
                'acc': acc,
                'racc': racc,
                'avg/loss': loss_meter.avg,
                'avg/cos-sim': cos_sim_meter.avg,
                'avg/acc': acc_meter.avg,
                'avg/racc': racc_meter.avg,
                'time/total-remaining': total_remaining,
                'time/this-epoch-remaining': this_epoch_remaining,
                'time/epoch-average-time': epoch_avg_time,
                'other/epoch': epoch + i / len(dataloader),
            }
            log_data.update(eval_logs)
            wandb.log(log_data, step=step_total)

        # save 10 models over training
        if args.save_checkpoints and (step_total % (args.steps // 10) == 0):
            torch.save(unwrap_model(model).model.state_dict(),
                       f'{args.output_dir}/checkpoints/step_{step_total}.pt')
            torch.save(optimizer.state_dict(),
                       f'{args.output_dir}/checkpoints/step_{step_total}_opt.pt')

        # every 20 steps, save a fallback model
        if step_total % 20 == 0:
            torch.save(unwrap_model(model).model.state_dict(),
                       f'{args.output_dir}/checkpoints/fallback_{step_total}.pt')
            torch.save(optimizer.state_dict(),
                       f'{args.output_dir}/checkpoints/fallback_{step_total}_opt.pt')
            # remove older fallback
            for file in os.listdir(f'{args.output_dir}/checkpoints'):
                if file.startswith('fallback') and str(step_total) not in file:
                    os.remove(f'{args.output_dir}/checkpoints/{file}')

        if step_total >= args.steps:
            break

        torch.cuda.empty_cache()

    return step_total


def evaluate(model, dataloader_eval, embedding_text_labels_norm, args):
    is_train = model.training
    model.eval()
    loss_eval_wrapper = ComputeLossWrapper(
        embedding_orig=None,
        embedding_text_labels_norm=embedding_text_labels_norm,
        reduction='none',
        loss='ce',
        logit_scale=100.,
    )

    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')
    cos_sim_meter = AverageMeter('cos-sim')

    for data_eval, targets_eval in dataloader_eval:
        data_eval, targets_eval = data_eval.cuda(), targets_eval.cuda()
        data_eval_adv = apgd(
            model=model,
            loss_fn=loss_eval_wrapper,
            x=data_eval,
            y=targets_eval,
            norm=args.norm,
            eps=args.eps,
            n_iter=50,
            initial_stepsize=0.05 * args.eps if args.clean_weight > 0 else None,
            verbose=False
        )
        with torch.no_grad():
            embedding_adv_eval_norm = model(data_eval_adv, output_normalize=True)
            logits_eval_adv = embedding_adv_eval_norm @ embedding_text_labels_norm
            racc_eval = compute_acc(logits_eval_adv, targets_eval)

            embedding_eval_norm = model(data_eval, output_normalize=True)
            logits_eval = embedding_eval_norm @ embedding_text_labels_norm
            acc_eval = compute_acc(logits_eval, targets_eval)

            cos_sim_eval = F.cosine_similarity(embedding_adv_eval_norm, embedding_eval_norm, dim=1).mean()

            acc_meter.update(acc_eval, data_eval.shape[0])
            racc_meter.update(racc_eval, data_eval.shape[0])
            cos_sim_meter.update(cos_sim_eval, data_eval.shape[0])

    acc_eval, racc_eval, cos_sim_eval = acc_meter.avg, racc_meter.avg, cos_sim_meter.avg
    logging.info(f'[eval-acc] {acc_eval:.2f} [eval-racc] {racc_eval:.2f} [eval-cos-sim] {cos_sim_eval:.3f} '
                 f'[n] {acc_meter.count}')
    if is_train:
        model.train()
    return {'eval/acc': acc_eval, 'eval/racc': racc_eval, 'eval/cos-sim': cos_sim_eval}


@torch.no_grad()
def compute_acc(logits, targets):
    preds_clean = logits.max(dim=1)[1].detach()
    acc = (preds_clean.eq(targets).sum() / targets.shape[0]).item() * 100
    return acc


def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'l1':
        loss = l1(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        loss = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction
        )
    elif loss_str == 'ce_reg':
        # ce + l2 embedding regularization
        loss_ce = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction
        )
        loss_l2 = l2(out=embedding, targets=embedding_orig, reduction=reduction)
        loss = 0.7 * loss_ce + 0.3 * loss_l2
    else:
        raise ValueError(f'loss {loss_str} not supported')
    return loss

def l2(out, targets, reduction='none'):
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), "Shapes mismatch in l2!"
    return squared_error_batch

def l1(out, targets, reduction='none'):
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    l1_error_batch = F.l1_loss(out, targets, reduction='none')
    if reduction == 'mean':
        l1_error_batch = torch.mean(l1_error_batch.sum(dim=1))
    else:
        l1_error_batch = l1_error_batch.sum(dim=1)
        assert l1_error_batch.shape == (out.shape[0],), "Shapes mismatch in l1!"
    return l1_error_batch

def ce(out, targets, reduction='mean'):
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    return F.cross_entropy(out, targets, reduction=reduction)


if __name__ == '__main__':
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse command-line arguments
    args = parser.parse_args()
    # Convert from 0-255 to 0-1
    args.eps /= 255
    args.stepsize_adv /= 255

    # Make sure eval_freq is a multiple of log_freq
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

    if args.devices != '':
        # set cuda visible devices
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    base_dir = "./"

    # Keep the same model name / directory if continuing
    if args.start_step == 0:
        # A fresh run: create new model name + random suffix
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        args.finetuned_model_name = f'{args.model_name}_{args.pretrained}_{args.dataset}_{args.loss}_{args.experiment_name}_{random_str}'
        args.finetuned_model_name = args.finetuned_model_name.replace('/', '_').replace(':', '_')
        # set output directory if user hasn't passed one
        if args.output_dir != '':
            args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
        else:
            args.output_dir = os.path.join(base_dir, "checkpoints", args.finetuned_model_name + "_temp")
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=False)
        logging.info(f"(Fresh run) Output dir: {args.output_dir}")
    else:
        # Resuming: do *not* re-randomize name or directory
        args.finetuned_model_name = args.optimizer_state.split('/')[-3]
        args.output_dir = os.path.join(base_dir, "checkpoints", args.finetuned_model_name)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
        logging.info(f"(Continuing run) Output dir: {args.output_dir}")

    # For record-keeping: store updated args
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    setup_logger(os.path.join(args.output_dir, "log"))
    logging.info(f"Output dir: {args.output_dir}")

    main(args)
