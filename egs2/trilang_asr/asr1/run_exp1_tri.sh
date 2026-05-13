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

stage=1
stop_stage=13
ngpu=1         # SERVER NOTE: set to 2 for DDP if running alone on server
               #   --ngpu 2 uses DDP; faster by ~1.7× but blocks 2 GPUs
nj=4
nbpe=5000

. utils/parse_options.sh || true

echo "=== Experiment 1: Trilingual Base Model (NO CS) ==="
echo "    ngpu=${ngpu}  stage=${stage}  stop_stage=${stop_stage}"

./asr.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu} \
    --nj ${nj} \
    --lang "trilingual" \
    --audio_format wav \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_train_text "data/tri/train/text" \
    --nlsyms_txt "local/nlsyms.txt" \
    --asr_config "conf/train_asr_conformer_s.yaml" \
    --inference_config "conf/decode_asr.yaml" \
    --use_lm false \
    --train_set "tri/train" \
    --valid_set "tri/dev" \
    --test_sets "id/test ar/test en/test cs/test" \
    --asr_tag "tri_base" \
    --gpu_inference true \
    "$@"
