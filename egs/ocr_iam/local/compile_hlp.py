#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script takes as input lang_dir and generates HLG from

    - H, the ctc topology, built from tokens contained in lang_dir/lexicon.txt
    - L, the lexicon, built from lang_dir/L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lang/lm/G_n_gram.fst.txt

The generated HLG is saved in $lang_dir/HLG.pt
"""
import argparse
import logging
from pathlib import Path

import k2
import torch

from icefall.lexicon import Lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm",
        type=str,
        default="P",
        help="""Stem name for LM used in HLG compiling.
        """,
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )

    return parser.parse_args()


def compile_HLP(lang_dir: str, lm: str = "G_3_gram") -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_5000.
      lm:
        The language stem base name.

    Return:
      An FSA representing HLG.
    """
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    H = k2.ctc_topo(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    logging.info(f"Loading {lm}.fst.txt")
    with open(f"{lang_dir}/{lm}.fst.txt") as f:
        P = k2.Fsa.from_openfst(f.read(), acceptor=False)
        torch.save(P.as_dict(), f"{lang_dir}/{lm}.pt")

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    L = k2.arc_sort(L)
    P = k2.arc_sort(P)

    logging.info("Intersecting L and P")
    LP = k2.compose(L, P)
    logging.info(f"LP shape: {LP.shape}")

    logging.info("Connecting LP")
    LP = k2.connect(LP)
    logging.info(f"LP shape after k2.connect: {LP.shape}")

    logging.info(type(LP.aux_labels))
    logging.info("Determinizing LP")

    LP = k2.determinize(LP)
    logging.info(type(LP.aux_labels))

    logging.info("Connecting LP after k2.determinize")
    LP = k2.connect(LP)

    logging.info("Removing disambiguation symbols on LP")

    # LG.labels[LG.labels >= first_token_disambig_id] = 0
    # see https://github.com/k2-fsa/k2/pull/1140
    labels = LP.labels
    labels[labels >= first_token_disambig_id] = 0
    LP.labels = labels

    assert isinstance(LP.aux_labels, k2.RaggedTensor)
    LP.aux_labels.values[LP.aux_labels.values >= first_word_disambig_id] = 0

    LP = k2.remove_epsilon(LP)
    logging.info(f"LP shape after k2.remove_epsilon: {LP.shape}")

    LP = k2.connect(LP)
    LP.aux_labels = LP.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LP")
    LP = k2.arc_sort(LP)

    logging.info("Composing H and LP")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    HLP = k2.compose(H, LP, inner_labels="tokens")
    HL = k2.compose(H, L, inner_labels="tokens")

    logging.info("Connecting LP")
    HLP = k2.connect(HLP)
    HL = k2.connect(HL)

    logging.info("Arc sorting LP")
    HLP = k2.arc_sort(HLP)
    HL = k2.arc_sort(HL)
    logging.info(f"HLP.shape: {HLP.shape}")

    return HLP, HL


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)


    logging.info(f"Processing {lang_dir}")

    HLP, HL = compile_HLP(lang_dir, args.lm)
    logging.info(f"Saving HLP.pt to {lang_dir}")
    torch.save(HLP.as_dict(), f"{lang_dir}/HLP.pt")
    torch.save(HL.as_dict(), f"{lang_dir}/HL.pt")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
