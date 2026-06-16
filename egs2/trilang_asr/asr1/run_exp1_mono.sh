#!/usr/bin/env bash
# run_exp1_mono.sh
# Experiment 1 — Monolingual baselines (ID, AR, EN)
# Wraps ESPnet2 asr.sh per language.
# Usage:
#   ./run_exp1_mono.sh --lang id [--stage 2] [--stop_stage 13]
#
# Server (RTX 2080 × 4, 2 parallel):
#   CUDA_VISIBLE_DEVICES=0 ./run_exp1_mono.sh --lang id &
#   CUDA_VISIBLE_DEVICES=1 ./run_exp1_mono.sh --lang ar &
#   wait
#   CUDA_VISIBLE_DEVICES=0 ./run_exp1_mono.sh --lang en

# set -e
# set -u
# set -o pipefail

lang=id       # id | ar | en
stage=2
stop_stage=13
ngpu=1
nj=4
inference_nj=2
nbpe=500

# NOTE: --langs must be first params, else ignored"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang)
            lang="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# NEW: Validate lang argument
case "$lang" in
  id|ar|en) ;;
  *) echo "ERROR: --lang must be id, ar, or en"; exit 1 ;;
esac


echo "=== Experiment 1: Monolingual ASR — ${lang^^} ==="
echo "    ngpu=${ngpu}  stage=${stage}  stop_stage=${stop_stage}"

# TODO: re-add  speed perturb later for faster train time 
# --speed_perturb_factors "0.9 1.0 1.1" \
./asr.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nj ${nj} \
    --inference_nj ${inference_nj} \
    --gpu_inference true \
    --ngpu ${ngpu} \
    --lang ${lang} \
    --audio_format wav \
    --feats_type raw \
    --min_wav_duration 1.0 \
    --max_wav_duration 30 \
    --token_type bpe \
    --nbpe ${nbpe} \
    --use_lm false \
    --asr_config "conf/train_asr_conformer.yaml" \
    --inference_config "conf/decode_asr.yaml" \
    --asr_tag "mono_${lang}_ctc5" \
    --train_set "${lang}/train" \
    --valid_set "${lang}/dev" \
    --test_sets "${lang}/test" \
    --bpe_train_text "data/${lang}/train/text" \
    --lm_train_text "data/${lang}/train/text" \
    --speed_perturb_factors "1.0" \
    "$@"
