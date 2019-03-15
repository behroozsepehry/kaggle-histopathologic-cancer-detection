import argparse
import yaml
import pandas as pd

from fastai import vision as fvision
from sklearn.metrics import roc_auc_score
import torch
from torchvision import models
import numpy as np


def train(settings):
    model_path = settings['Logger']['args']['log_dir']
    path = settings['Dataloaders']['path']
    train_folder = f'{path}train'
    test_folder = f'{path}test'
    train_lbl = f'{path}train_labels.csv'
    ORG_SIZE = 96

    bs = 64
    num_workers = 4  # Apprently 2 cpus per kaggle node, so 4 threads I think
    sz = 96

    df_trn = pd.read_csv(train_lbl)

    tfms = fvision.get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,
                          max_lighting=0.05, max_warp=0.)

    data = fvision.ImageDataBunch.from_csv(path, csv_labels=train_lbl, folder='train', ds_tfms=tfms, size=sz, suffix='.tif',
                                   test=test_folder, bs=bs);

    stats = data.batch_stats()
    data.normalize(stats)

    data.show_batch(rows=5, figsize=(12, 9))

    def auc_score(y_pred, y_true, tens=True):
        score = roc_auc_score(y_true, torch.sigmoid(y_pred)[:, 1])
        if tens:
            score = fvision.tensor(score)
        else:
            score = score
        return score

    learn = fvision.create_cnn(
        data,
        models.densenet201,
        path=model_path,
        metrics=[auc_score],
        ps=0.5
    )
    learn.lr_find()
    learn.recorder.plot()

    lr = 1e-04

    learn.fit_one_cycle(1, lr)
    learn.recorder.plot()
    learn.recorder.plot_losses()

    learn.unfreeze()
    learn.lr_find()

    learn.fit_one_cycle(10, slice(1e-4, 1e-3))
    learn.recorder.plot()
    learn.recorder.plot_losses()

    preds, y = learn.get_preds()
    pred_score = auc_score(preds, y)
    pred_score

    preds, y = learn.TTA()
    pred_score_tta = auc_score(preds, y)
    pred_score_tta

    preds_test, y_test = learn.get_preds(ds_type=fvision.DatasetType.Test)
    preds_test_tta, y_test_tta = learn.TTA(ds_type=fvision.DatasetType.Test)
    sub = pd.read_csv(f'{path}/sample_submission.csv').set_index('id')
    sub.head()
    clean_fname = np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])
    fname_cleaned = clean_fname(data.test_ds.items)
    fname_cleaned = fname_cleaned.astype(str)

    sub.loc[fname_cleaned, 'label'] = fvision.to_np(preds_test[:, 1])
    sub.to_csv(f'submission_{pred_score}.csv')

    sub.loc[fname_cleaned, 'label'] = fvision.to_np(preds_test_tta[:, 1])
    sub.to_csv(f'submission_{pred_score_tta}.csv')


def main():
    parser = argparse.ArgumentParser(description='MNIST classification')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/cancer.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        settings = yaml.load(f)

    if settings['function'] == 'train':
        result = train(settings)
    else:
        raise NotImplementedError
    print('result: %s' % (result))


if __name__ == '__main__':
    main()