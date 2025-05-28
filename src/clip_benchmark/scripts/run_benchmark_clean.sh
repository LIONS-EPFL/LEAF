#!/bin/bash
set -e # stop on error
# add parent to python path
#export PYTHONPATH="../":"${PYTHONPATH}"

SECONDS=0
SAMPLES=1000
BS=1000

SAVE_DIR="/mnt/cschlarmann37/project_bimodal-robust-clip/results_zeroshot_img_class/${SAMPLES}smpls"  # TODO
mkdir -p "$SAVE_DIR"
python -m clip_benchmark.cli eval --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
--dataset src/clip_benchmark/benchmark/datasets.txt \
--pretrained_model src/clip_benchmark/benchmark/models.txt \
--output "${SAVE_DIR}/clean_{model}_{pretrained}_beta{beta}_{dataset}_{n_samples}_bs{bs}_{attack}_{eps}_{iterations}.json" \
--attack none --eps 1 \
--batch_size $BS --n_samples $SAMPLES \
#--model_type open_clip # TODO


hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo "[Runtime] $hours h $minutes min"