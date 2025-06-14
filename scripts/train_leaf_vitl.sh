python3 train_AT_text_only.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --wandb-project-name "datacomp_small" \
    --train-data 'path/to/datacomp/shards/{00000000..00001287}.tar' \
    --imagenet-val='path/to/imagenet/val' \
    --val-text-classification 'fancyzhx/ag_news' \
    --warmup 1400 \
    --batch-size=128 \
    --accum-freq=1 \
    --lr=1e-5 \
    --wd=1e-4 \
    --epochs=30 \
    --workers=8 \
    --model 'hf-hub:chs20/fare2-clip' \
    --dataset-type webdataset \
    --train-num-samples 80000 \
    --val-num-samples 1024 \
    --k_adv 1 \
    --k_adv_test 1 \
    --rho=50 \
    --n_charmer_test=20 \
    --n_val_imagenet 1000 \
    --seed 1 \
    --custom_out_folder 'ViT-L-FARE2_constrained_' \
    --constrain \