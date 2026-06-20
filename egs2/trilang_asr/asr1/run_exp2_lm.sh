#!/usr/bin/env bash
# set -e
# set -u
# set -o pipefail

stage=6 # start directly to training LM
skip_stages="10 11 " # skip ASR training; reuse from exp1
stop_stage=13
ngpu=1
nj=4
lm_weight=0.3
nbpe=1000

tri_model="exp/asr_tri_base/valid.acc.best.pth"
if [ ! -f "${tri_model}" ]; then
    echo "ERROR: Trilingual base model not found: ${tri_model}"
    echo "Run run_exp1_tri.sh first (stages 1-13)."
    exit 1
fi

echo "=== Experiment 2: Shallow Fusion (opt-350m, lm_weight=${lm_weight}) ==="

./asr.sh \
    --stage ${stage} \
    --skip_stages ${skip_stages} \
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
    --inference_args "--lm_weight ${lm_weight}" \
    --asr_tag "tri_lm{lm_weight}" \
    --train_set "tri/train" \
    --valid_set "tri/dev" \
    --test_sets "id/test ar/test en/test cs/test" \
    --asr_model_file "${tri_model}" \
    --bpe_train_text "data/tri/train/text" \
    --lm_train_text "data/tri/train/text" \
    --speed_perturb_factors "1.0" \
    "$@"
