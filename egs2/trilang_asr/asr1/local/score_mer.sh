#!/bin/bash

# Copyright 2026 ESPnet Contributors
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Scoring script for Mixed Error Rate (MER) evaluation
# For trilingual code-switched ASR (AR, EN, ID)

. ./path.sh || exit 1;

# defaults
nj=1
cmd=run.pl
lang=ar_en_id
ref_channel=
stage=0
stop_stage=1

# parse options
. utils/parse_options.sh

if [ $# -lt 3 ]; then
    echo "Usage: $0 <lang> <data-dir> <exp-dir>"
    echo "E.g.: $0 ar_en_id data/test exp/asr_model/decode_test"
    echo ""
    echo "options:"
    echo "  --nj <nj>                # number of parallel jobs"
    echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
    echo "  --lang <lang>            # language combination (ar_en_id | ar_en | en_id, etc.)"
    echo "  --stage <stage>          # stage to start (default=0)"
    echo "  --stop_stage <stop_stage> # stage to stop (default=1)"
    exit 1;
fi

lang=$1
data=$2
expdir=$3

# Parse language codes
IFS='_' read -ra LANGS <<< "$lang"
languages=$(printf ',%s' "${LANGS[@]}" | cut -c 2-)
logging_info="Languages: ${languages}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Calculate MER for: $lang"
    echo "Ref:  $data/text"
    echo "Hyp:  $expdir/1best_recog/text"
    echo "Dir:  $expdir/MER"
    
    [ ! -f $data/text ] && echo "Error: no such file $data/text" && exit 1;
    [ ! -f $expdir/1best_recog/text ] && echo "Error: no such file $expdir/1best_recog/text" && exit 1;
    
    mkdir -p $expdir/MER
    
    python3 pyscripts/utils/evaluate_mer.py \
        $data/text \
        $expdir/1best_recog/text \
        --outdir $expdir/MER \
        --languages $languages \
        --lang_delim "<" \
        --verbose 1
    
    echo "MER evaluation completed. Results in: $expdir/MER/"
fi
