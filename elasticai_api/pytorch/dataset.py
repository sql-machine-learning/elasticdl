# Copyright 2021 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Any, Tuple

import torchvision
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class ElasticImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
    ):
        """Create a dataset from a folder for ElasticDL
        Arguments:
            root: the path of the image folder
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
                loader (callable): A function to load a sample given its path.
            is_valid_file (callable, optional): A function that takes path of
                a file and check if the file is a valid file (used to check of
                corrupt files) both extensions and is_valid_file should not
                be passed.
        """
        super(ElasticImageFolder, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self._data_shard_service = None

    def set_data_shard_service(self, data_shard_service):
        self._data_shard_service = data_shard_service

    def __len__(self):
        if self._data_shard_service:
            # Set the maxsize because the size of dataset is not fixed
            # when using dynamic sharding
            return sys.maxsize
        else:
            return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is
                class_index of the target class.
        """
        if self._data_shard_service:
            index = self._data_shard_service.fetch_record_index()
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
