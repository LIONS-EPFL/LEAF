import json
import logging
import os
import sys
import time
import types
import numpy as np
from transformers import CLIPModel

import open_clip
import torch
import torch.nn.functional as F
from torchvision import transforms
import wandb
import argparse
from robustbench import benchmark
from robustbench.data import load_clean_dataset
from autoattack import AutoAttack
from robustbench.model_zoo.enums import BenchmarkDataset
from eval_utils import compute_accuracy_no_dataloader, load_clip_model
#from ..train.utils import str2bool
from open_clip import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, CLIP
#from ..train.log import setup_logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError


parser = argparse.ArgumentParser(description="Script arguments")

parser.add_argument('--model_name', type=str, default='none', help='ViT-L-14, ViT-B-32, don\'t use if wandb_id is set')
parser.add_argument('--pretrained', type=str, default='none', help='Pretrained model ckpt path, don\'t use if wandb_id is set')
parser.add_argument('--wandb_id', type=str, default='none', help='Wandb id of training run, don\'t use if model_name and pretrained are set')
parser.add_argument('--logit_scale', type=str2bool, default=True, help='Whether to scale logits')
parser.add_argument('--full_benchmark', type=str2bool, default=False, help='Whether to run full RB benchmark')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
parser.add_argument('--cifar10_root', type=str, default='/mnt/datasets/CIFAR10', help='CIFAR10 dataset root directory')
parser.add_argument('--cifar100_root', type=str, default='/mnt/datasets/CIFAR100', help='CIFAR100 dataset root directory')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_samples_imagenet', type=int, default=1000, help='Number of samples from ImageNet for benchmark')
# parser.add_argument('--n_samples_cifar', type=int, default=1000, help='Number of samples from CIFAR for benchmark')
parser.add_argument('--template', type=str, default='ensemble', help='Text template type; std, ensemble')
parser.add_argument('--norm', type=str, default='linf', help='Norm for attacks; linf, l2')
parser.add_argument('--eps', type=float, default=2., help='Epsilon for attack')
# parser.add_argument('--alpha', type=float, default=2., help='APGD alpha parameter')
parser.add_argument('--experiment_name', type=str, default='/mnt/cschlarmann37/project_bimodal-robust-clip/clip-adversarial-images/', help='Experiment name for logging')
parser.add_argument('--blackbox_only', type=str2bool, default=False, help='Run blackbox attacks only')
parser.add_argument('--save_images', type=str2bool, default=True, help='Save images during benchmarking')
parser.add_argument('--wandb', type=str2bool, default=False, help='Use Weights & Biases for logging')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')


CIFAR10_LABELS = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ClassificationModel(torch.nn.Module):
    def __init__(self, model, text_embedding, args, input_normalize, resizer=None, logit_scale=True):
        super().__init__()
        self.model = model
        self.args = args
        self.input_normalize = input_normalize
        self.resizer = resizer if resizer is not None else lambda x: x
        self.text_embedding = text_embedding
        self.logit_scale = logit_scale

    def forward(self, image, output_normalize=True):
        assert output_normalize

        if isinstance(self.model, CLIP):  # open_clip
            embedding_norm_ = self.model.encode_image(
                self.input_normalize(self.resizer(image)),
                normalize=True
            )
        elif isinstance(self.model, CLIPModel):  # hf clip
            embedding_norm_ = self.model.get_image_features(self.input_normalize(self.resizer(image)))
            embedding_norm_ = embedding_norm_ / embedding_norm_.norm(p=2, dim=-1, keepdim=True)

        logits = embedding_norm_ @ self.text_embedding
        if self.logit_scale:
            logits *= self.model.logit_scale.exp()
        return logits

def encode_text_wrapper_CLIPModel(self, x, normalize = False):
    out = self.get_text_features(x)
    if normalize:
        out = out / torch.norm(out,dim=-1,keepdim=True)
    return out

def main(args):
    # print args
    logging.info(f"Arguments:")
    logging.info("\n".join(f"{arg}: {value}" for arg, value in vars(args).items()))

    args.eps /= 255
    # make sure there is no string in args that should be a bool
    assert not any(
        [isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values(
        )]
    )

    if args.dataset == 'imagenet':
        num_classes = 1000
        data_dir = args.imagenet_root
        n_samples = args.n_samples_imagenet
        resizer = None
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    eps = args.eps

    # init wandb
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    wandb_user, wandb_project = 'chs20', 'clip-finetune'
    while True:
        try:
            run_eval = wandb.init(
                project=wandb_project,
                job_type='eval',
                name=f'{"rb" if args.full_benchmark else "aa"}-clip-{args.dataset}-{args.norm}-{eps:.2f}'
                     f'-{args.wandb_id if args.wandb_id is not None else args.pretrained}-{args.blackbox_only}',
                save_code=True,
                config=vars(args),
                mode='online' if args.wandb else 'disabled'
            )
            break
        except wandb.errors.CommError as e:
            logging.warning('wandb connection error', file=sys.stderr)
            logging.warning(f'error: {e}', file=sys.stderr)
            time.sleep(1)
            logging.info('retrying..', file=sys.stderr)

    if args.devices != '':
        # set cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    main_device = 0
    num_gpus = torch.cuda.device_count()
    print('GPUS:',num_gpus)
    if num_gpus > 1:
        logging.info(f"Number of GPUs available: {num_gpus}")
    else:
        logging.info("No multiple GPUs available.")

    if not args.blackbox_only:
        attacks_to_run = ['apgd-ce', 'apgd-t']
    else:
        attacks_to_run = ['square']
    logging.info(f'[attacks_to_run] {attacks_to_run}')

    if args.wandb_id not in [None, 'none', 'None']:
        assert args.pretrained in [None, 'none', 'None']
        assert args.model_name in [None, 'none', 'None']
        api = wandb.Api()
        run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
        model_name = run_train.config['model_name']
        logging.info(f'model_name: {model_name}')
        pretrained = run_train.config["output_dir"]
        if pretrained.endswith('_temp'):
            pretrained = pretrained[:-5]
        pretrained += "/checkpoints/final.pt"
    else:
        model_name = args.model_name
        pretrained = args.pretrained
        run_train = None
    del args.model_name, args.pretrained

    logging.info(f'[loading pretrained clip] {model_name} {pretrained}')

    model, preprocessor_without_normalize, normalize = load_clip_model(model_name, pretrained)
    # tokenizer = open_clip.get_tokenizer(model_name)
    logging.warning("using default tokenizer")
    tokenizer = open_clip.tokenize  # TODO

    if args.dataset != 'imagenet':
        assert False
        # make sure we don't resize outside the model as this influences threat model
        preprocessor_without_normalize = transforms.ToTensor()
    logging.info(f'[resizer] {resizer}')
    logging.info(f'[preprocessor] {preprocessor_without_normalize}')
    logging.info(f'[normalizer] {normalize}')

    model.eval()
    model.to(main_device)
    # model.encode_text = types.MethodType(encode_text_wrapper_CLIPModel, model)
    
    with torch.no_grad():
        embedding_text_labels_norm = build_zero_shot_classifier(
            model=model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=50,
            device=f'cuda:{main_device}',
            use_tqdm=True
        )

        assert torch.allclose(
            F.normalize(embedding_text_labels_norm, dim=0),
            embedding_text_labels_norm
        )
        logging.info(f'[text embeddings shape] {embedding_text_labels_norm.shape}')

    # get model
    model = ClassificationModel(
        model=model,
        text_embedding=embedding_text_labels_norm,
        args=args,
        resizer=resizer,
        input_normalize=normalize,
        logit_scale=args.logit_scale,
    )

    if num_gpus > 1:
        raise NotImplementedError
    model = model.cuda()
    model.eval()

    device = torch.device(main_device)
    torch.cuda.empty_cache()

    dataset_short = (
        'img' if args.dataset == 'imagenet' else
        'c10' if args.dataset == 'cifar10' else
        'c100' if args.dataset == 'cifar100' else
        'unknown'
    )

    start = time.time()
    if args.full_benchmark:
        raise NotImplementedError
    else:
        adversary = AutoAttack(
            model, norm=args.norm.replace('l', 'L'), eps=eps, version='custom', attacks_to_run=attacks_to_run,
            # alpha=args.alpha,
            verbose=True
        )

        x_test, y_test = load_clean_dataset(
            BenchmarkDataset(args.dataset), n_examples=n_samples, data_dir=data_dir,
            prepr=preprocessor_without_normalize, )

        acc = compute_accuracy_no_dataloader(
            model, data=x_test, targets=y_test, device=device, batch_size=args.batch_size
            ) * 100
        logging.info(f'[acc] {acc:.2f}%')
        x_adv, y_adv = adversary.run_standard_evaluation(
            x_test, y_test, bs=args.batch_size, return_labels=True
            )  # y_adv are preds on x_adv
        racc = compute_accuracy_no_dataloader(
            model, data=x_adv, targets=y_test, device=device, batch_size=args.batch_size
            ) * 100
        logging.info(f'[acc] {acc:.2f}% [racc] {racc:.2f}%')

        # save
        model_name_clean = model_name.replace("/", "-").replace(" ", "-")
        pretrained_clean = pretrained.split('/')[-2].replace("/", "-").replace(" ", "-") if pretrained not in ['none', 'None'] else 'none'
        # save adv image
        res_dir = os.path.join(args.experiment_name, f'{args.dataset}/{model_name_clean}-{pretrained_clean}-{args.norm}-{eps:.3f}-'
                         # f'alph{args.alpha:.3f}-'
                         f'{n_samples}smpls-{time.strftime("%Y-%m-%d_%H-%M-%S")}')
        os.makedirs(args.experiment_name, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        logging.info(f'[saving results to] {res_dir}')
        if args.save_images:
            # save the adversarial images
            x_adv = x_adv.detach().cpu()
            y_adv = y_adv.detach().cpu()
            x_clean = x_test.detach().cpu()
            y_clean = y_test.detach().cpu()
            torch.save(x_adv, f'{res_dir}/x_adv.pt')
            torch.save(y_adv, f'{res_dir}/y_adv.pt')
            torch.save(x_clean, f'{res_dir}/x_clean.pt')
            torch.save(y_clean, f'{res_dir}/y_clean.pt')
        with open(f'{res_dir}/args.json', 'w') as f:
            json.dump(vars(args), f)
        with open(f'{res_dir}/results.json', 'w') as f:
            json.dump({'acc': acc, 'racc': racc}, f)

        # write to wandb
        if run_train is not None:
            # reload the run to make sure we have the latest summary
            del api, run_train
            api = wandb.Api()
            run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
            if args.dataset == 'imagenet':
                assert args.norm == 'linf'
                eps_descr = str(int(eps * 255))
                if eps_descr == '4':
                    descr = dataset_short
                else:
                    descr = f'{dataset_short}-eps{eps_descr}'
                if n_samples != 5000:
                    acc = f'{acc:.2f}*'
                    racc = f'{racc:.2f}*'
            elif args.dataset == 'cifar10':
                if args.norm == 'linf':
                    descr = dataset_short
                else:
                    descr = f'{dataset_short}-{args.norm}'
                if n_samples != 10000:
                    acc = f'{acc:.2f}*'
                    racc = f'{racc:.2f}*'
            else:
                raise ValueError(f'Unknown dataset: {args.dataset}')
            run_train.summary.update({f'aa/acc-{dataset_short}': acc})
            run_train.summary.update({f'aa/racc-{descr}': racc})
            run_train.summary.update()
            run_train.update()
    run_eval.finish()


if __name__ == '__main__':
    # python -m CLIP_eval.clip_robustbench --model_name none --pretrained none --wandb_id ID --dataset imagenet --norm linf --eps 4
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse command-line arguments
    args = parser.parse_args()
    #setup_logger(log_file=None)

    main(args)





