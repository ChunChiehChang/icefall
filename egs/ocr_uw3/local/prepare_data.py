#!/usr/bin/env python3

"""
This file downloads the uw3 data.
"""
import os
import pickle
import random
import tarfile

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from lhotse.utils import Pathlike, safe_extract

def download_uw3(
    target_dir: Pathlike = "data/download",
    source_dir: Pathlike = "/path/to/uw3/dataset",
) -> Path:
    """
    """
    target_dir = Path(target_dir)
    source_dir = Path(source_dir)
    
    lines_tar_path = source_dir / "uw3-lines-book.tgz"

    if not os.path.isdir(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(lines_tar_path) as tar:
            safe_extract(tar, path=target_dir / "uw3")

    return target_dir

# TODO: Include extra corpus data
# Not real lhotse manifests, just named similarly for ease of understanding
# Trying to avoid modifying lhotse as it doesn't support images
def prepare_uw3(
    download_dir: Pathlike = "data/download",
    output_dir: Pathlike = "data/manifests",
):
    """
    """
    download_dir = Path(download_dir)
    download_dir = download_dir / 'uw3' / 'book'
    output_dir = Path(output_dir)

    random.seed(0)
    page_count = 0
    train_images = []
    test_images = []
    full_text = []
    full_text_test = []
    lexicon = {
        '<SIL>':'SIL',
        '<UNK>':'SIL',
    }
    for page in sorted(os.listdir(download_dir)):
        page_path = download_dir / page
        page_count = page_count + 1
        for line in sorted(os.listdir(page_path)):
            if line.endswith('.txt'):
                text_path = page_path / line
                gt_fh = open(text_path, 'r')
                text = gt_fh.readlines()[0].strip()

                image_name = line.split('.')[0]
                image_path = page_path / (image_name + ".png")
                image_id = page + '_' + image_name

                data_element = {
                    "image_id": image_id,
                    "writer_id": page_count,
                    "image_path": image_path,
                    "text": text,
                }
                # The UW3 dataset doesn't have established training and testing splits
                # The dataset is randomly split train 95% and test 5%
                coin = random.randint(0, 20)
                if coin >= 1:
                    train_images.append(data_element)
                    full_text.append(text)
                    for word in text.split():
                        if not word in lexicon:
                            lexicon[word] = ' '.join(word)
                elif coin < 1:
                    test_images.append(data_element)
                    full_text_test.append(text)

    if output_dir is not None:
        with open(output_dir / f"uw3_images_train.pkl", 'wb') as f:
            pickle.dump(train_images, f)
        with open(output_dir / f"uw3_images_test.pkl", 'wb') as f:
            pickle.dump(test_images, f)
        with open(output_dir / f"uw3_train_text.txt", 'w') as f:
            for line in full_text:
                f.write(line + '\n')
        with open(output_dir / f"uw3_test_text.txt", 'w') as f:
            for line in full_text_test:
                f.write(line + '\n')
        with open(output_dir / f"uw3_train_lexicon.txt", 'w') as f:
            for key, value in sorted(lexicon.items()):
                f.write(key + ' ' + value.replace('#', '<HASH>') + '\n')

