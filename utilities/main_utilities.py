import os
import numpy as np
import pandas as pd
from torchvision.utils import save_image

import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets import folder as dataset_folder

from utilities import general_utilities
from utilities import nn_utilities


def get_instance(folder, instance_type, **kwargs):
    kwargs_instance = kwargs
    instance_filename = kwargs_instance.get('name')
    if not instance_filename:
        instance = None
    else:
        module = general_utilities.import_from_path(folder+instance_filename+'.py')
        instance = getattr(module, instance_type)(**kwargs_instance)
    return instance


def get_model(**kwargs):
    return get_instance('models/', 'Model', **kwargs)


def get_optimizer(parameters, **kwargs):
    optimizer_name = kwargs['name']
    optimizer_constructor = getattr(optim, optimizer_name)
    optimizer = optimizer_constructor(parameters, **kwargs['args'])
    return optimizer


def get_lr_scheduler(optimizer, **kwargs):
    scheduler_name = kwargs['name']
    scheduler_constructor = getattr(optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_constructor(optimizer, **kwargs['args'])
    return scheduler


def get_dataloaders(**kwargs):
    path = kwargs.get('path')
    path_train = os.path.join(path, 'train')
    path_test = os.path.join(path, 'test')

    data = pd.read_csv(os.path.join(path, 'train_labels.csv'))
    train_df = data.set_index('id')
    train_keys = train_df.index.values
    train_labels = np.asarray(train_df['label'].values)
    train_labels_dict = {train_keys[i]: train_labels[i] for i in range(len(train_keys))}

    general_utilities.create_labeled_dataset_folder(path_train, train_labels_dict)

    transforms_dict = {}
    for phase in ['train', 'test', 'val']:
        transforms_dict[phase] = [getattr(transforms, t['name'])(**t.get('args', {}))
                                  for t in kwargs.get('transforms', {}).get(phase, [])] + [transforms.ToTensor()]

    dataset_train = datasets.DatasetFolder(path_train, loader=dataset_folder.default_loader, extensions=['tif'],
                                           transform=transforms.Compose(transforms_dict['train']),
                                           target_transform=lambda xxx: torch.FloatTensor([xxx]))
    dataset_val = datasets.DatasetFolder(path_train, loader=dataset_folder.default_loader, extensions=['tif'],
                                         transform=transforms.Compose(transforms_dict['val']),
                                         target_transform=lambda xxx: torch.FloatTensor([xxx]))
    dataset_test = nn_utilities.ImageFolderWithPaths(path_test, transform=transforms.Compose(transforms_dict['test']))

    dataset_train_size = len(dataset_train)
    subset_samplers = nn_utilities.get_subset_dataset_sampler(kwargs['ratio'], range(dataset_train_size))

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(
                dataset_train,
                sampler=subset_samplers['train'],
                **kwargs['args']),
        'val':
            torch.utils.data.DataLoader(
                dataset_val,
                sampler=subset_samplers['val'],
                **kwargs['args']),
        'test':
            torch.utils.data.DataLoader(
                dataset_test,
                **dict(kwargs['args'], shuffle=False))
    }
    return dataloaders


def get_loss(**kwargs):
    loss_name = kwargs['name']
    loss_constructor = getattr(nn, loss_name)
    loss_func = loss_constructor(**kwargs.get('args', {}))
    return loss_func


def get_device(**kwargs):
    device_name = kwargs['name']
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device_name_2 = 'cpu'
        device = torch.device(device_name_2)
        if device_name_2 != device_name:
            print('Warning: device \'%s\' not available, using device \'%s\' instead'% (device_name, device_name_2))
    return device


def get_logger(**kwargs):
    if kwargs['name'] == 'tensorboard':
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(**kwargs['args'])
        logger.flags = kwargs.get('flags', {})
    elif not kwargs['name']:
        logger = None
    else:
        raise NotImplementedError
    return logger


if __name__ == '__main__':
    import argparse
    import yaml
    import os
    os.chdir('..')
    parser = argparse.ArgumentParser(description='Cancer classification')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/cancer.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        settings = yaml.load(f)

    dataloaders = get_dataloaders(**settings['Dataloaders'])
    for x, y in dataloaders['train']:
        save_image(x, settings['Dataloaders']['path'] + 'sample_train.png')
        print('train labels:\n%s' % y)
        break
    for x, y in dataloaders['val']:
        save_image(x, settings['Dataloaders']['path'] + 'sample_val.png')
        print('val labels:\n%s' % y)
        break
    for x, p in dataloaders['test']:
        save_image(x, settings['Dataloaders']['path'] + 'sample_test.png')
        print('test paths:\n%s' % list(p))
        break


