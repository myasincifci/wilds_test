import os
from typing import Any, List
from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import random


def get_datasets(
        root_dir: str, 
        domains: List, 
        train=True,
        split_rat: List[int]=[10_000, 1_000], 
        transforms=None
):
    # Read in metadata
    _metadata_df = pd.read_csv(
        os.path.join(root_dir, 'metadata.csv'),
        index_col=0,
        dtype={'patient': 'str'}
    )

    _root_dir = root_dir
    _domains = domains

    # Get filenames
    _inputs = {node: [] for node in list(set(_metadata_df.loc[:, ["node"]].values.flatten()))}
    for patient, node, x, y, tumor in _metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord', 'tumor']].itertuples(index=False, name=None):
        _inputs[node].append((f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png', tumor))

    lengths = {k: len(_inputs[k]) for k in _inputs}

    samples = {node: [] for node in _domains}
    for k in _domains:
        samples[k] = random.sample(_inputs[k], sum(split_rat))

    train_set = {node: [] for node in _domains}
    for k in _domains:
        train_set[k] = samples[k][0:split_rat[0]]

    test_set = {node: [] for node in _domains}
    for k in _domains:
        test_set[k] = samples[k][split_rat[0]:]

    transform = T.Compose([
      T.ToTensor()
    ])

    train_dataset, test_dataset = WildsDataset(train_set, root_dir, transform), WildsDataset(test_set, root_dir, transform)


    return train_dataset, test_dataset

class WildsDataset(Dataset):
    def __init__(
            self,
            split,
            root_dir,
            transforms
        ) -> None:

        self._split = split 
        self._transforms = transforms
        self._root_dir = root_dir

    def __len__(self) -> int:
        return list(self._split.values())[0].__len__()

    def __getitem__(self, index) -> Any:
        domains = list(self._split.keys())

        img_filenames = [os.path.join(self._root_dir,self._split[d][index][0]) for d in domains]
        imgs = list(Image.open(f).convert("RGB") for f in img_filenames)
        targets = [self._split[d][index][1] for d in domains]

        if self._transforms:
            imgs = [self._transforms(img) for img in imgs]

        return imgs, targets

if __name__== "__main__":
    train_set, test_set = get_datasets(root_dir="./data/camelyon17_v1.0", domains=[0, 4], train=False, split_rat=[90_000, 10_000], transforms=T.ToTensor())

    transform = T.Compose([
      T.ToTensor()
    ])

    train_dataset, test_dataset = WildsDataset(train_set, "./data/camelyon17_v1.0", transform), WildsDataset(test_set, "./data/camelyon17_v1.0", transform)

    train_dataset.__getitem__(0)

    pass