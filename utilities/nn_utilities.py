import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


def apply_func_to_model_data(model, func, dataloader, device):
    result = []
    with torch.no_grad():
        for i_batch, data_tuple in enumerate(dataloader):
            x = data_tuple[0]
            x = x.to(device)
            model_out = model(x)
            result.append(func(model_out, data_tuple))
    return result


def data_parallel_model(model, input, ngpu):
    if 'cuda' in str(input.device) and ngpu > 1:
        output = nn.parallel.data_parallel(model, input, range(ngpu))
    else:
        output = model(input)
    return output


def get_subset_dataset_sampler(ratios, idxs):
    """
    :param ratio: dict, values should sum to one
    :param idxs: range
    :return: dict key: ratio keys; value: sampler
    """
    idxs = np.array(idxs)
    dataset_size = len(idxs)
    np.random.shuffle(idxs)
    dataset_samplers = {}
    split_idx = 0
    for phase, ratio in ratios.items():
        next_split_idx = int(np.floor(split_idx + ratio * dataset_size))
        # if last phase
        if len(dataset_samplers) == len(ratios) - 1:
            next_split_idx = None
        dataset_samplers[phase] = SubsetRandomSampler(idxs[split_idx:next_split_idx])
        split_idx = next_split_idx
    return dataset_samplers


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        """
         Args:
             index (int): Index

         Returns:
             tuple: (image, target) where target is class_index of the target class.
         """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, path

    def __len__(self):
        return len(self.imgs)

