#!/usr/bin/env bash
# set -e
# set -u
# set -o pipefail

# ---- phase toggles (turn off whichever you don't need to re-run) -----------
run_data_prep=true
# run_data_prep=false
run_finetune=true
run_finetune_lm=false
run_decode_nolm=true
run_decode_lm=true

approach="tri"

ngpu=1
nj=4
inference_nj=2
lm_weight=0.3
speed_perturb_factors="1.0"

nbpe_cs=1000 # 500 or 1k

tri_model="exp/asr_tri_base_bpe1000_lr5e4_warm15k_epoch100/valid.acc.ave_10best.pth"

asr_tag="tri_cs_ft_bpe${nbpe_cs}_epoch10"
# asr_tag="tri_cs_bpe${nbpe_cs}_epoch60"

# test using trilingual lm opt instead of re-tuned on CS data
cs_lm_exp="exp/lm_train_lm_opt_trilingual_bpe${nbpe_cs}"
cs_train="cs_${approach}/train"
cs_dev="cs_${approach}/dev"
cs_test="cs_${approach}/test"


if [ "${run_data_prep}" = true ]; then
    # retrain cs with all tri+cs+cs_edge data
    utils/combine_data.sh "data/cs_tri/train" \
        "data/cs_edge/train" \
        "data/cs/train" \
        "data/tri/train"
    utils/combine_data.sh "data/cs_tri/dev" \
        "data/cs_edge/dev" \
        "data/cs/dev" \
        "data/tri/dev"
    utils/combine_data.sh "data/cs_tri/test" \
        "data/cs_edge/test" \
        "data/cs/test" \
        "data/tri/test"
fi

if [ ! -f "${tri_model}" ]; then
    echo "ERROR: Trilingual base model not found: ${tri_model}"
    echo "Run run_exp1_tri.sh first."
    exit 1
fi

# TODO: changes approach pt.2
if [ ! -d "data/${cs_train}" ]; then
    echo "  INFO: data/${cs_train} not found, falling back to shared data/cs/"
    cs_train="cs/train"
    cs_dev="cs/dev"
    cs_test="cs/test"
fi

test_sets_all="cs/test id/test ar/test en/test"

echo "=== Experiment 3: CS Fine-tuning (Approach ${approach}) ==="

# -----------------------------------------------------------------------------
# Stage 1 — CS data prep + fresh 500-BPE or 1k-BPE token list (asr.sh stages 2-5)
# Must carry the same audio/feat/bpe flags Phase 1 uses below: stage 5 is where
# the sentencepiece model gets trained (on cs_train text), and the
# dump/ layout built here is what Phase 1 picks up when it resumes at stage 10.
# -----------------------------------------------------------------------------
if [ "${run_data_prep}" = true ]; then
    echo "    Stage A: CS data prep + ${nbpe_cs}-BPE token list"
    ./asr.sh \
        --stage 2 \
        --stop_stage 5 \
        --nj ${nj} \
        --ngpu ${ngpu} \
        --lang "trilingual_cs" \
        --audio_format wav \
        --feats_type raw \
        --min_wav_duration 1.0 \
        --max_wav_duration 30 \
        --token_type bpe \
        --nbpe ${nbpe_cs} \
        --speed_perturb_factors ${speed_perturb_factors} \
        --train_set "${cs_train}" \
        --valid_set "${cs_dev}" \
        --test_sets "${test_sets_all}" \
        --bpe_train_text "data/${cs_train}/text" \
        --lm_train_text "data/${cs_train}/text"

    echo
    echo "=== CS data preparation done! ==="
fi

# -----------------------------------------------------------------------------
# Phase 1a — Fine-tune from tri_base (stages 10-11: ASR stats + training only)
# use_lm is false here on purpose: the external LM is shallow-fused at DECODE
# time only (Phase 2b), never jointly trained with the AM — see notebook
# Step 2/3 for the --pretrained_model + --ignore_init_mismatch pattern.
# -----------------------------------------------------------------------------
if [ "${run_finetune}" = true ]; then
    echo "    Phase 1a: Fine-tuning on CS data (stages 10-11)"
    echo "    pretrained_model=${tri_model}"
    echo
    echo

    # re-add the following cmd args for finetuning prevs model
    # --pretrained_model "${tri_model}" \
    ./asr.sh \
        --stage 10 \
        --stop_stage 11 \
        --nj ${nj} \
        --ngpu ${ngpu} \
        --lang "trilingual" \
        --audio_format wav \
        --feats_type raw \
        --token_type bpe \
        --nbpe ${nbpe_cs} \
        --use_lm false \
        --ignore_init_mismatch true \
        --asr_config "conf/train_asr_conformer_tri.yaml" \
        --asr_tag ${asr_tag} \
        --train_set "${cs_train}" \
        --valid_set "${cs_dev}" \
        --test_sets "${test_sets_all}" \
        --lm_train_text "data/${cs_train}/text" \
        --speed_perturb_factors ${speed_perturb_factors} \
        "$@"
fi

# -----------------------------------------------------------------------------
# Phase 1b — re-finetune LM with CS (stages 6-9)
# -----------------------------------------------------------------------------
if [ "${run_finetune_lm}" = true ]; then
    echo "    Phase 1b: Fine-tuning lm on CS data (stages 10-11)"
    echo "    pretrained_model=${tri_model}"
    echo
    echo

    # train lm on both tri and cs data
    cat "data/${cs_train}/text" > "data/${cs_train}/../text"
    cat "data/tri/train/text" >> "data/${cs_train}/../text"

    # adjust lm_conf if lm are being finetuned 
    ./asr.sh \
        --stage 6 \
        --stop_stage 9 \
        --nj ${nj} \
        --ngpu ${ngpu} \
        --lang "trilingual_cs" \
        --audio_format wav \
        --feats_type raw \
        --token_type bpe \
        --nbpe ${nbpe_cs} \
        --use_lm true \
        --lm_conf "conf/train_lm_opt.yaml" \
        --ignore_init_mismatch true \
        --asr_config "conf/train_asr_conformer_tri.yaml" \
        --asr_tag ${asr_tag} \
        --train_set "${cs_train}" \
        --valid_set "${cs_dev}" \
        --test_sets "${test_sets_all}" \
        --lm_train_text "data/${cs_train}/../text" \
        --speed_perturb_factors ${speed_perturb_factors} \
        "$@"
fi

# -----------------------------------------------------------------------------
# Phase 2a — Decode without LM (stages 12-13)
# -----------------------------------------------------------------------------
if [ "${run_decode_nolm}" = true ]; then
    echo "    Phase 2a: Decoding without LM"
    echo
    echo

    ./asr.sh \
        --stage 12 \
        --stop_stage 13 \
        --nj ${nj} \
        --inference_nj ${inference_nj} \
        --gpu_inference true \
        --ngpu ${ngpu} \
        --lang "trilingual_cs" \
        --token_type bpe \
        --nbpe ${nbpe_cs} \
        --use_lm false \
        --inference_config "conf/decode_asr.yaml" \
        --asr_tag ${asr_tag} \
        --train_set "${cs_train}" \
        --valid_set "${cs_dev}" \
        --test_sets "${test_sets_all}" \
        --speed_perturb_factors ${speed_perturb_factors} \
        "$@"
fi

# -----------------------------------------------------------------------------
# Phase 2b — Decode WITH shallow-fused LM (stages 12-13)
# Only runs once cs_lm_exp points at an LM trained on the matching 500-token
# CS BPE. --lm_exp tells asr.sh to reuse an already-trained LM checkpoint
# instead of retraining (see espnet2_tutorial.md: "--skip_train true --asr_exp
# <dir> --lm_exp <dir>" pattern for reusing existing experiment artefacts).
# -----------------------------------------------------------------------------
if [ "${run_decode_lm}" = true ]; then
    if [ -n "${cs_lm_exp}" ]; then
        echo "    Phase 2b: Decoding with shallow-fused LM (lm_exp=${cs_lm_exp}, lm_weight=${lm_weight})"
        echo
        echo

        ./asr.sh \
            --stage 12 \
            --stop_stage 13 \
            --nj ${nj} \
            --inference_nj ${inference_nj} \
            --gpu_inference true \
            --ngpu ${ngpu} \
            --lang "trilingual_cs" \
            --token_type bpe \
            --nbpe ${nbpe_cs} \
            --use_lm true \
            --lm_exp "${cs_lm_exp}" \
            --inference_config "conf/decode_asr.yaml" \
            --inference_args "--lm_weight ${lm_weight}" \
            --inference_lm "latest.pth" \
            --asr_tag ${asr_tag} \
            --train_set "${cs_train}" \
            --valid_set "${cs_dev}" \
            --test_sets "${test_sets_all}" \
            --speed_perturb_factors ${speed_perturb_factors} \
            "$@"

    else
        echo "    Phase 2b SKIPPED: cs_lm_exp is not set."
        echo "    Fine-tune/replace the OPT LM's vocab layers on the ${nbpe_cs}-token CS BPE first"
        echo "    (same ignore-mismatch pattern as the AM), then set cs_lm_exp and rerun."
    fi
fi
