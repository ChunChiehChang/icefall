#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

dataset_dir=/home/hltcoe/cchang/download/UW3
download_dir=data/download
manifest_dir=data/manifests

lang_dir=data/lang
lm_dir=data/lm

. shared/parse_options.sh || exit 1

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $download_dir"
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Download data"
    python -c """
import local.prepare_data as prepare_data
prepare_data.download_uw3('${download_dir}','${dataset_dir}')
"""
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare IAM manifest"
    mkdir -p ${manifest_dir}
    python -c """
import local.prepare_data as prepare_data
prepare_data.prepare_uw3('${download_dir}','${manifest_dir}')
"""
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Prepare HL"
    mkdir -p ${lang_dir}
    cat ${manifest_dir}/uw3_train_lexicon.txt | sort > ${lang_dir}/lexicon.txt
    ./local/prepare_lang.py
    ./local/prepare_lang_fst.py --lang-dir ${lang_dir} --has-silence 1
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare G"
    mkdir -p ${lang_dir}/lm
    if [ ! -f data/lang/lm/G_3_gram.fst.txt ]; then
        ./shared/make_kn_lm.py \
            -ngram-order 3 \
            -text ${lang_dir}/lexicon.txt \
            -lm ${lang_dir}/lm/3-gram.arpa
        python3 -m kaldilm \
            --read-symbol-table="${lang_dir}/words.txt" \
            --disambig-symbol='#0' \
            --max-order=3 \
            ${lang_dir}/lm/3-gram.arpa > ${lang_dir}/lm/G_3_gram.fst.txt
    fi

    if [ ! -f data/lang/lm/G_4_gram.fst.txt ]; then
        ./shared/make_kn_lm.py \
            -ngram-order 4 \
            -text ${lang_dir}/lexicon.txt \
            -lm ${lang_dir}/lm/4-gram.arpa
        python3 -m kaldilm \
            --read-symbol-table="${lang_dir}/words.txt" \
            --disambig-symbol='#0' \
            --max-order=4 \
            ${lang_dir}/lm/4-gram.arpa > ${lang_dir}/lm/G_4_gram.fst.txt
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Compile HLG"
    if [ ! -f $lang_dir/HLG.pt ]; then
        ./local/compile_hlg.py --lang-dir ${lang_dir}
    fi
fi
