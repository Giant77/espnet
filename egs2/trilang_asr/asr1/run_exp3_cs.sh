#!/usr/bin/env bash
# run_exp3_cs.sh
# Experiment 3 — CS fine-tuning from trilingual base
# Trains TWO fine-tuned models: one per CS data generation approach (A and B).
# Per global resolution: CS data used ONLY here, not in initial training.
#
# SERVER NOTE:
#   approach=A and approach=B can run in parallel on separate GPUs:
#     CUDA_VISIBLE_DEVICES=0 ./run_exp3_cs.sh --approach A &
#     CUDA_VISIBLE_DEVICES=1 ./run_exp3_cs.sh --approach B &
#     wait

# set -e
# set -u
# set -o pipefail

approach=A     # A (TTS synthetic) | B (audio concatenation)
               # data/cs/train must contain the appropriate CS data
stage=1
stop_stage=13
ngpu=1
nj=4
nbpe=2000

. utils/parse_options.sh || true

case "$approach" in
  A|B) ;;
  *) echo "ERROR: --approach must be A or B"; exit 1 ;;
esac

tri_model="exp/asr_tri_base/valid.acc.best.pth"
if [ ! -f "${tri_model}" ]; then
    echo "ERROR: Trilingual base model not found: ${tri_model}"
    echo "Run run_exp1_tri.sh first."
    exit 1
fi

echo "=== Experiment 3: CS Fine-tuning (Approach ${approach}) ==="
echo "    ngpu=${ngpu}  stage=${stage}  init_param=${tri_model}"

# CS data dir: data/cs_approach_A/ or data/cs_approach_B/
# Create appropriate symlink or dir before running
cs_train="cs_${approach}/train"
cs_dev="cs_${approach}/dev"
cs_test="cs_${approach}/test"

# Verify CS data exists
if [ ! -d "data/${cs_train}" ]; then
    # Fallback to shared cs/ directory
    cs_train="cs/train"
    cs_dev="cs/dev"
    cs_test="cs/test"
    echo "  INFO: Using shared data/cs/ for approach ${approach}"
fi

./asr.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu} \
    --nj ${nj} \
    --lang "trilingual_cs" \
    --audio_format wav \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_train_text "data/tri/train/text" \
    --nlsyms_txt "local/nlsyms.txt" \
    --asr_config "conf/finetune_asr_cs.yaml" \
    --inference_config "conf/decode_asr.yaml" \
    --use_lm false \
    --train_set "${cs_train}" \
    --valid_set "${cs_dev}" \
    --test_sets "${cs_test} id/test ar/test en/test" \
    --asr_tag "cs_finetune_${approach}" \
    --gpu_inference true \
    --asr_args "--init_param ${tri_model}" \
    "$@"
