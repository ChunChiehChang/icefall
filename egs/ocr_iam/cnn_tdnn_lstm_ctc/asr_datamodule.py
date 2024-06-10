# Copyright      2021  Piotr Żelasko
#                2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import List

from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures

import local.iam_dataset as iam_dataset
from torch.utils.data import DataLoader

from icefall.dataset.datamodule import DataModule
from icefall.utils import str2bool


class IAMOCRDataModule(DataModule):
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--feature-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Path to directory with pickle.",
        )
        group.add_argument(
            "--image-height",
            type=int,
            default=40,
            help="Image loading Height",
        )

    def train_dataloaders(self) -> DataLoader:
        logging.info("About to create train dataset")

        train_config = iam_dataset.IAMConfig(
            scale_height=self.args.image_height,
            augment={'rotation':0,'shear':10},
            #augment={'rotation':0,'shear':10, 'resample':True, 'resized_crop':True},
            pickle=[self.args.feature_dir/'iam_images_train.pkl'],
            train=True,
            batch_size=16,
        )
        train = iam_dataset.IAMDataset(train_config)

        logging.info("About to create train dataloader")
        train_dl = DataLoader(
            train,
            sampler=train.sampler,
            batch_size=train_config.batch_size,
            collate_fn=train.collate_fn,
            drop_last=True,
        )

        return train_dl

    def test_dataloaders(self) -> DataLoader:
        logging.debug("About to create test dataset")
        test_config = iam_dataset.IAMConfig(
            scale_height=self.args.image_height,
            augment=None,
            pickle=[self.args.feature_dir/'iam_images_test.pkl'],
            train=False,
            batch_size=16,
        )
        test = iam_dataset.IAMDataset(test_config)
        
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            sampler=test.sampler,
            batch_size=test_config.batch_size,
            collate_fn=test.collate_fn,
        )
        return test_dl

    def valid_dataloaders(self) -> DataLoader:
        logging.debug("About to create dev dataset")
        dev_config = iam_dataset.IAMConfig(
            scale_height=self.args.image_height,
            augment=None,
            pickle=[self.args.feature_dir/'iam_images_dev.pkl'],
            train=False,
            batch_size=16,
        )
        dev = iam_dataset.IAMDataset(dev_config)

        logging.debug("About to create dev dataloader")
        dev_dl = DataLoader(
            dev,
            sampler=dev.sampler,
            batch_size=dev_config.batch_size,
            collate_fn=dev.collate_fn,
        )
        return dev_dl

