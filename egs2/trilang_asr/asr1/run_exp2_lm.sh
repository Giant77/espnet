#!/usr/bin/env bash
# run_exp2_lm.sh
# Experiment 2 — Trilingual + XGLM-564M shallow fusion
# Decodes the ALREADY TRAINED tri_base model with LM enabled.
# Run AFTER run_exp1_tri.sh completes (requires exp/tri_base/).
#
# NOTE: LM weight (lm_weight=0.3) is a starting point;
#       grid search on dev set recommended (0.1, 0.2, 0.3, 0.4, 0.5)
# SERVER NOTE: Decoding only — ngpu=1 sufficient; can run 2 decodes in parallel

# set -e
# set -u
# set -o pipefail

stage=12       # Start directly from decoding — ASR model reused from exp1
stop_stage=13
ngpu=1
nj=4
lm_weight=0.3  # tuned on dev; override with --lm_weight X
nbpe=1000

tri_model="exp/asr_tri_base/valid.acc.best.pth"
if [ ! -f "${tri_model}" ]; then
    echo "ERROR: Trilingual base model not found: ${tri_model}"
    echo "Run run_exp1_tri.sh first (stages 1-13)."
    exit 1
fi

echo "=== Experiment 2: Shallow Fusion (XGLM-564M, lm_weight=${lm_weight}) ==="

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
    --use_lm true \
    --token_type bpe \
    --nbpe ${nbpe} \
    --asr_config "conf/train_asr_conformer_s.yaml" \
    --inference_config "conf/decode_asr_lm.yaml" \
    --inference_args "--lm_weight ${lm_weight}" \
    --asr_tag "tri_lm_w${lm_weight}" \
    --train_set "tri/train" \
    --valid_set "tri/dev" \
    --test_sets "id/test ar/test en/test cs/test" \
    --lm_config "conf/train_lm_xglm.yaml" \
    --asr_model_file "${tri_model}" \
    --download_model XGLM-564M \
    --speed_perturb_factors "1.0" \
    "$@"
