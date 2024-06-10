#!/usr/bin/env python3

"""
This file downloads the iam data.
"""
import os
import pickle
import tarfile
import torchvision
import xml.dom.minidom as minidom
import zipfile

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from lhotse.utils import Pathlike, safe_extract

def download_iam(
    target_dir: Pathlike = "data/download",
    source_dir: Pathlike = "/path/to/iam/dataset",
) -> Path:
    """
    """
    target_dir = Path(target_dir)
    source_dir = Path(source_dir)
    
    lines_tar_path = source_dir / "lines.tgz"
    xml_tar_path = source_dir / "xml.tgz"
    ascii_tar_path = source_dir / "ascii.tgz"
    datasplit_zip_path = source_dir / "largeWriterIndependentTextLineRecognitionTask.zip"

    if not os.path.isdir(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(lines_tar_path) as tar:
            safe_extract(tar, path=target_dir / "lines")
        with tarfile.open(xml_tar_path) as tar:
            safe_extract(tar, path=target_dir / "xml")
        with tarfile.open(ascii_tar_path) as tar:
            safe_extract(tar, path=target_dir / "ascii")
        with zipfile.ZipFile(datasplit_zip_path) as zip_f:
            zip_f.extractall(target_dir / "largeWriterIndependentTextLineRecognitionTask")

    return target_dir

# TODO: Include extra corpus data
# Not real lhotse manifests, just named similarly for ease of understanding
# Trying to avoid modifying lhotse as it doesn't support images
def prepare_iam(
    download_dir: Pathlike = "data/download",
    output_dir: Pathlike = "data/manifests",
) -> Dict[str, List[Dict[str, str]]]:
    """
    """

    download_dir = Path(download_dir)
    output_dir = Path(output_dir)

    gt_path = download_dir / "ascii" / "lines.txt"
    gt_dict = {}
    with open(gt_path, 'rt') as gt_file:
        for line in gt_file:
            if line[0] == '#':
                continue
            line = line.strip()
            utt_id = line.split(' ')[0]
            text_vect = line.split(' ')[8:]
            text = "".join(text_vect)
            text = text.replace('|', ' ')
            gt_dict[utt_id] = text

    manifests = defaultdict(dict)
    for dataset in ['train', 'dev', 'test']:
        uttlist = []
        supervision_path = download_dir / "largeWriterIndependentTextLineRecognitionTask"
        if dataset == 'dev':
            uttlist.append(supervision_path / "validationset1.txt")
            uttlist.append(supervision_path / "validationset2.txt")
        else:
            uttlist.append(supervision_path / (dataset + "set.txt"))

        images, full_text, lexicon = _prepare_dataset(uttlist, download_dir, gt_dict)
        
        if output_dir is not None:
            with open(output_dir / f"iam_images_{dataset}.pkl", 'wb') as f:
                pickle.dump(images, f)

            with open(output_dir / f"iam_{dataset}_text.txt", 'w') as f:
                for line in full_text:
                    f.write(line + '\n')
            with open(output_dir / f"iam_{dataset}_lexicon.txt", 'w') as f:
                for key, value in lexicon.items():
                    f.write(key + ' ' + value.replace('#', '<HASH>') + '\n')

        manifests[dataset] = {
            "images": images
        }
    return manifests

def _prepare_dataset(
    dataset_list: List[Pathlike],
    download_dir: Pathlike,
    gt_dict: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    """
    image_list = []
    full_text = []
    lexicon = {}
    lexicon['<SIL>'] = 'SIL'
    lexicon['<UNK>'] = 'SIL'
    for dataset in dataset_list:
        with open(dataset) as f:
            for line in f:
                line = line.strip()
                line_vect = line.split('-')
                xml_file = line_vect[0] + '-' + line_vect[1]
                xml_path = download_dir / f"xml/{xml_file}.xml"
                img_num = line[-3:]
                doc = minidom.parse(str(xml_path))
                form_elements = doc.getElementsByTagName('form')[0]
                writer_id = form_elements.getAttribute('writer-id')
                outerfolder = form_elements.getAttribute('id')[0:3]
                innerfolder = form_elements.getAttribute('id')
                image_path = download_dir / f"lines/{outerfolder}/{innerfolder}/{innerfolder}{img_num}.png"
                text = gt_dict[line]
                image_id = line
                image = torchvision.io.read_image(str(image_path))
                image_size = torchvision.transforms.functional.get_image_size(image)
                image_length = image_size[0] / image_size[1]
                data_element = {
                    "image_id": image_id,
                    "image_length": image_length,
                    "writer_id": writer_id,
                    "image_path": image_path,
                    "text": text
                }
                image_list.append(data_element)
                full_text.append(text)
                for word in text.split():
                    if not word in lexicon:
                        lexicon[word] = ' '.join(word)
                        for letter in word:
                            if not letter in lexicon:
                                lexicon[letter] = letter
    return image_list, full_text, lexicon
