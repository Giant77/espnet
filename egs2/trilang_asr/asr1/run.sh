#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
set -o pipefail

stage=1
stop_stage=100


lang=id # id, en, ar, cs

# Data directories
data_dir="data"

# Feature settings
feats_type=fbank
sample_rate=16000

# Conf
asr_config=conf/train_asr_conformer_s.yaml

# LM settings
use_lm=false         # set true for Experiment 2 (shallow fusion)
lm_config=conf/train_lm_xglm.yaml
lm_weight=0.3        # tuned on dev set

# Decoding
decode_config=conf/decode_asr.yaml

# Paths
asr_stats_dir="exp/asr_stats"
ngpu=1
nj=4                 # parallel jobs — set to CPU core count

. utils/parse_options.sh || true

# Experiment names
id_tag=mono_id
ar_tag=mono_ar
en_tag=mono_en
tri_tag=trilingual_base
cs_tag=cs_finetune   # fine-tuned from tri model


# train_set=train_"$(echo "${lang}" | tr - _)"
# train_dev=dev_"$(echo "${lang}" | tr - _)"
# test_set="${train_dev} test_$(echo ${lang} | tr - _)"

# nlsyms_txt=data/nlsyms.txt
# monolingual_asr_config=conf/train_asr.yaml
# multilingual_asr_config=conf/tuning/train_asr_conformer_hier_lid_utt.yaml
# lm_config=conf/train_lm.yaml
# inference_config=conf/decode_lid.yaml

# if [[ "zh" == *"${lang}"* ]]; then
#   nbpe=2500
# elif [[ "fr" == *"${lang}"* ]]; then
#   nbpe=350
# elif [[ "es" == *"${lang}"* ]]; then
#   nbpe=235
# elif [[ "all" == *"${lang}"* ]]; then
#   nbpe=6500
# else
#   nbpe=300
# fi

# if [[ "all" == *"${lang}"* ]]; then
#   ./asr.sh \
#       --lang "${lang}" \
#       --auxiliary_data_tags "lid_utt " \
#       --local_data_opts "--stage 0 --lang ${lang} --nlsyms_txt ${nlsyms_txt}" \
#       --post_process_local_data_opts "--stage 2 --lang ${lang} --nlsyms_txt ${nlsyms_txt}" \
#       --audio_format "wav" \
#       --use_lm false \
#       --feats_normalize utt_mvn \
#       --lm_config "${lm_config}" \
#       --token_type bpe \
#       --nbpe $nbpe \
#       --bpe_nlsyms "${nlsyms_txt}" \
#       --feats_type raw \
#       --speed_perturb_factors "0.9 1.0 1.1" \
#       --asr_config "${multilingual_asr_config}" \
#       --inference_config "${inference_config}" \
#       --train_set "${train_set}" \
#       --valid_set "${train_dev}" \
#       --test_sets "${test_set}" \
#       --bpe_train_text "data/${train_set}/text" \
#       --local_score_opts "--score_lang_id true" "$@" \
#       --lm_train_text "data/${train_set}/text" "$@"
# else
#   ./asr.sh \
#       --lang "${lang}" \
#       --local_data_opts "--lang ${lang}" \
#       --audio_format "wav" \
#       --use_lm false \
#       --feats_normalize utt_mvn \
#       --lm_config "${lm_config}" \
#       --token_type bpe \
#       --nbpe $nbpe \
#       --feats_type raw \
#       --speed_perturb_factors "0.9 1.0 1.1" \
#       --asr_config "${monolingual_asr_config}" \
#       --inference_config "${inference_config}" \
#       --train_set "${train_set}" \
#       --valid_set "${train_dev}" \
#       --test_sets "${test_set}" \
#       --bpe_train_text "data/${train_set}/text" \
#       --lm_train_text "data/${train_set}/text" \
#       --local_score_opts "--score_lang_id false" "$@"
# fi


# ==============================

train_set="id/train"
valid_set="id/dev"
test_sets="id/test"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang id \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

