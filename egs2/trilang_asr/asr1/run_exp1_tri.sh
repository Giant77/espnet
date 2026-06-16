#!/usr/bin/env bash
# run_exp1_tri.sh
# Experiment 1 — Trilingual base model (ID+AR+EN, NO CS data)
# Per global resolution: CS data intentionally excluded here.
#
# Server (RTX 2080, single run full GPU):
#   Conformer-S (d_model=256):  ~5-6GB → fits one 8GB card → ngpu=1
#   Conformer-M (d_model=512):  ~7-8GB → fits one 8GB card → ngpu=1
#   For 2x GPU DDP: --ngpu 2 (see SERVER NOTE below)
#
# Server parallel strategy:
#   Run mono models first (run_exp1_mono.sh × 3 on 3 GPUs),
#   then run this on the remaining GPU.

# set -e
# set -u
# set -o pipefail

stage=2
stop_stage=13
ngpu=1
nj=4
inference_nj=2
nbpe=1000

echo "=== Experiment 1: Trilingual Base Model (NO CS) ==="
echo "    ngpu=${ngpu}  stage=${stage}  stop_stage=${stop_stage}"

# train LM, but skip usage (lm_weight: 0); usage on exp2
./asr.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nj ${nj} \
    --inference_nj ${inference_nj} \
    --gpu_inference true \
    --ngpu ${ngpu} \
    --lang "trilingual" \
    --audio_format wav \
    --feats_type raw \
    --min_wav_duration 1.0 \
    --max_wav_duration 30 \
    --token_type bpe \
    --nbpe ${nbpe} \
    --use_lm true \
    --lm_config "conf/train_lm_opt.yaml" \
    --asr_config "conf/train_asr_conformer_tri.yaml" \
    --inference_config "conf/decode_asr.yaml" \
    --asr_tag "tri_base" \
    --train_set "tri/train" \
    --valid_set "tri/dev" \
    --test_sets "id/test ar/test en/test cs/test" \
    --bpe_train_text "data/tri/train/text" \
    --lm_train_text "data/tri/train/text" \
    --speed_perturb_factors "1.0" \
    "$@"
