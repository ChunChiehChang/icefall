#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

dataset_dir=/home/hltcoe/cchang/download/IAM/
download_dir=data/download
manifest_dir=data/manifests

lang_dir=data/lang
lang_bpe=${lang_dir}_bpe
vocab_size=500

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
prepare_data.download_iam('${download_dir}','${dataset_dir}')
"""
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare IAM manifest"
    python -c """
import local.prepare_data as prepare_data
prepare_data.prepare_iam('${download_dir}','${manifest_dir}')
"""
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Prepare BPE"
    mkdir -p ${lang_dir}
    mkdir -p ${lang_bpe}
    cat ${manifest_dir}/iam_train_lexicon.txt | sort -u > ${lang_dir}/lexicon.txt
#    cat ${manifest_dir}/iam_train_lexicon.txt ${manifest_dir}/iam_dev_lexicon.txt ${manifest_dir}/iam_test_lexicon.txt | \
#        sort -u > ${lang_dir}/lexicon.txt
    cat ${manifest_dir}/iam_train_text.txt > ${lang_dir}/iam.txt
#    cat ${manifest_dir}/iam_train_text.txt ${manifest_dir}/iam_dev_text.txt ${manifest_dir}/iam_test_text.txt \
#        > ${lang_dir}/iam.txt

    ./local/prepare_lang.py --lang-dir ${lang_dir}

    cp ${lang_dir}/words.txt ${lang_bpe}/words.txt
    cp ${lang_dir}/iam.txt ${lang_bpe}/iam.txt
    ./local/train_bpe_model.py \
        --lang-dir ${lang_bpe} \
        --vocab-size $vocab_size \
        --transcript ${lang_bpe}/iam.txt

    ./local/prepare_lang_bpe_wip.py --lang-dir ${lang_bpe}

    log "Validating ${lang_bpe}/lexicon.txt"
    ./local/validate_bpe_lexicon.py \
        --lexicon ${lang_bpe}/lexicon.txt \
        --bpe-model ${lang_bpe}/bpe.model

    log "Converting L.pt to L.fst"
    ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        ${lang_bpe}/L.pt \
        ${lang_bpe}/L.fst

    log "Converting L_disambig.pt to L_disambig.fst"
    ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        ${lang_bpe}/L_disambig.pt \
        ${lang_bpe}/L_disambig.fst
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare G"
    mkdir -p ${lang_dir}/lm
    ./local/convert_transcript_words_to_tokens.py \
        --lexicon ${lang_bpe}/lexicon.txt \
        --transcript ${lang_bpe}/iam.txt \
        --oov "<unk>" \
        > ${lang_bpe}/iam_tokens.txt

    ./shared/make_kn_lm.py \
        -ngram-order 6 \
        -text ${lang_bpe}/iam_tokens.txt \
        -lm ${lang_bpe}/P.arpa

    python3 -m kaldilm \
        --read-symbol-table="${lang_bpe}/words.txt" \
        --disambig-symbol='#0' \
        --max-order=6 \
        ${lang_bpe}/P.arpa > ${lang_bpe}/P.fst.txt
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Compile HLG"
    ./local/compile_hlp.py --lang-dir ${lang_bpe}
fi
