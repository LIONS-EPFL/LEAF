export PYTHONPATH="$PYTHONPATH:./src"

python src/robust_vlm/train/adversarial_training_clip.py \
--model_name hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K --dataset imagenet \
--imagenet_root /path/to/imagenet/ \
--template ensemble --output_normalize False \
--steps 10000 --warmup 700 --batch_size 128 \
--loss l2 --opt adamw --lr 1e-5 --wd 1e-4 --attack pgd \
--inner_loss l2 --norm linf --eps 2 \
--iterations_adv 10 --stepsize_adv 1 \
--wandb True --experiment_name FARE2 --log_freq 10