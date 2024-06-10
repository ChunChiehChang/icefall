import numpy as np
import pickle
import random
import torch
import torchvision

from image.transforms import get_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler


class IAMConfig():
    def __init__(
        self,
        augment={'rotation':0,'shear':0, 'resample':True, 'resize_and_crop':True},
        grayscale=1,
        scale_height=60,
        pickle=["data/manifests/iam_images_train.pkl"],
        padding=20,
        gaussian_noise=0.001,
        resized_crop=True,
        train=True,
        batch_size=16,
    ):
        super().__init__()
        self.augment = augment
        self.grayscale = grayscale
        self.scale_height = scale_height
        self.pickle = pickle
        self.padding = padding
        self.gaussian_noise = gaussian_noise
        self.train = train
        self.batch_size = batch_size

class IAMCollator(object):
    def __init__(self, config):
        self.image_height = config.scale_height
        self.channels = config.grayscale

    def __call__(self, batch):
        image_id = [item['supervisions']['image_id'] for item in batch]
        sequence_idx = [index for index, item in enumerate(batch)]
        writer_id = [item['supervisions']['writer_id'] for item in batch]
        image_path = [item['supervisions']['image_path'] for item in batch]
        text = [item['supervisions']['text'] for item in batch]
        start_frame = [item['supervisions']['start_frame'] for item in batch]
        width = [item['supervisions']['num_frames'] for item in batch]

        input_type = type(batch[0]['inputs'])
        inputs = torch.ones([len(batch), batch[0]['inputs'].shape[0], max(width), self.image_height]).type(input_type)
        for idx, item in enumerate(batch):
            inputs[idx, :, :item['supervisions']['num_frames'], :] = item['inputs']
        
        supervisions = {
            'image_id': image_id,
            'sequence_idx': torch.Tensor(sequence_idx),
            'start_frame': torch.Tensor(start_frame),
            'num_frames': torch.Tensor(width),
            'writer_id': writer_id,
            'image_path': image_path,
            'text': text,
        }
        item = {
            'inputs': inputs,
            'supervisions': supervisions
        }
        return item

class IAMDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.image_list = []
        for pkl in config.pickle:
            with open(pkl, 'rb') as f:
                self.image_list = self.image_list + pickle.load(f)
        self.image_aug_list = []
        if config.augment:
            for entry in self.image_list:
                aug_entry = entry.copy()
                aug_entry['image_id'] = aug_entry['image_id'] + '_aug'
                self.image_aug_list.append(aug_entry)
            self.image_list = self.image_list + self.image_aug_list 
            self.transforms = get_transforms(self.config)
        self.dataset_length = len(self.image_list)
        self.preprocess_only = get_transforms(self.config, preprocess_only=True)
        self.collate_fn = IAMCollator(self.config)
        self.sampler = IAMSampler(self.image_list, self.dataset_length, train=config.train, batch_size=config.batch_size)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        entry = self.image_list[index]
        image = torchvision.io.read_image(str(entry['image_path']), mode=torchvision.io.ImageReadMode.GRAY) / 255
        
        # A couple random images are rotated 90%
        if image.shape[1] > image.shape[2]:
            image = torch.transpose(image, 2, 1)

        # Apply Transforms or Preprocessing
        if entry['image_id'].endswith('_aug'):
            image = self.transforms(image)
        else:
            image = self.preprocess_only(image)
        image = torch.transpose(image, 2, 1)

        supervisions = {
            'image_id': entry['image_id'],
            'start_frame': 0,
            'num_frames': image.shape[1],
            'writer_id': entry['writer_id'],
            'image_path': entry['image_path'],
            'text': entry['text'],
        }
        return {'inputs':image, 'supervisions':supervisions}

class IAMSampler(Sampler):
    def __init__(self, image_list, dataset_length, train=True, batch_size=16, buckets=20):
        self.image_list = image_list
        self.dataset_length = dataset_length
        self.epoch = 0
        self.rng = random.Random(self.epoch)
        self.train = train
        self.batch_size = batch_size
        self.buckets = buckets
        self.indices = list(range(self.dataset_length))
        self.indices.sort(key=self.sort_fn)
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng.seed(self.epoch)
    def sort_fn(self, entry):
        return self.image_list[entry]['image_length']
    def __iter__(self):
        shuffled_indices0 = []
        shuffled_indices1 = []
        buckets = np.array_split(self.indices, self.buckets)
        for bucket in buckets:
            self.rng.shuffle(bucket)
            shuffled_indices0 = shuffled_indices0 + list(bucket)
        batches = [shuffled_indices0[i:(i+self.batch_size)] for i in range(0, self.dataset_length, self.batch_size)]
        self.rng.shuffle(batches)
        for batch in batches:
            shuffled_indices1 = shuffled_indices1 + batch
        return iter(shuffled_indices1)

    def __len__(self):
        return self.dataset_length

