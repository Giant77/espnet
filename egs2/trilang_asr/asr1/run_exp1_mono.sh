#!/usr/bin/env bash
# run_exp1_mono.sh
# Experiment 1 — Monolingual baselines (ID, AR, EN)
# Wraps ESPnet2 asr.sh per language.
# Usage:
#   ./run_exp1_mono.sh --lang id [--stage 1] [--stop_stage 13]
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

nbpe=1000

echo "=== Experiment 1: Monolingual ASR — ${lang^^} ==="
echo "    ngpu=${ngpu}  stage=${stage}  stop_stage=${stop_stage}"

./asr.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu} \
    --nj ${nj} \
    --lang ${lang} \
    --audio_format wav \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_train_text "data/${lang}/train/text" \
    --nlsyms_txt "local/nlsyms.txt" \
    --asr_config "conf/train_asr_conformer_s.yaml" \
    --inference_config "conf/decode_asr.yaml" \
    --use_lm false \
    --train_set "${lang}/train" \
    --valid_set "${lang}/dev" \
    --test_sets "${lang}/test" \
    --asr_tag "mono_${lang}" \
    --gpu_inference true \
    --expdir "exp_bpe${nbpe}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    "$@"
