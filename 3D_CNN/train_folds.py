import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
import h5py
from tqdm import tqdm
import random
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from create_model import create_nn
from dataset import MRIDataset
from utils import seed_everything, score, w_nae_score
from utils import fit_epoch, eval_epoch, train, make_val_preds


MRI_TR = 'data/fMRI_train/'
MRI_TE = 'data/fMRI_test/'
WEIGHTS_DIR = 'model_weights'
HISTORY_DIR = 'train_history'

if not os.path.isdir(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)
if not os.path.isdir(HISTORY_DIR):
    os.mkdir(HISTORY_DIR)


LR = 0.00103
BS = 16
N_EPOCHS = 15
N_FOLDS = 5
N_CLASSES = 1
SEED = 111
SAVE_HISTORY = False
seed_everything(SEED)
folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


train_data = pd.read_csv('data/train_scores.csv')
TARGETS = train_data.columns[1:]
for targ in TARGETS:
    train_data[targ].fillna((train_data[targ].mean()+train_data[targ].median())/2,
                            inplace=True)


def kfold_separate_targets():
    '''
    This function takes no arguments.
    Performs "train_kfold" function
    on separated targets.
    Returns data frame of predicted validation data.
    '''
    oof_df = pd.DataFrame(data=np.zeros(train_data[TARGETS].shape), columns=TARGETS)
    for targ in TARGETS:
        train_kfold(targ, oof_df, SAVE_HISTORY)

    return oof_df


def train_kfold(targ, oof_df, save_history=False):
    '''
    This function performs kfold models training
    on target and saves results.
    :param targ: target.
    :param oof_df: data frame for validation predictions.
    :param save_history: bool, save history.json or not.
    :return: none
    '''
    print(f'Target : {targ}')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data, train_data)):
        print(f'Fold {fold_+1}')

        tr_data = train_data.loc[trn_idx]
        val_data = train_data.loc[val_idx]

        tr_dataset = MRIDataset(MRI_TR, tr_data, 'train', targ)
        val_dataset = MRIDataset(MRI_TR, val_data, 'val', targ)

        model = create_nn(N_CLASSES)
        opt = torch.optim.SGD(model.parameters(), lr=LR,
                              momentum=0.9, weight_decay=7e-4)
        criterion = nn.L1Loss()

        with torch.autograd.set_detect_anomaly(True):
            history = train(tr_dataset, val_dataset, model, epochs=N_EPOCHS,
                            batch_size=BS, opt=opt, criterion=criterion)

        if save_history:
            loss, acc, val_loss, val_acc = zip(*history)
            history_dict = {
                'train_loss': loss,
                'val_loss': val_loss,
                'train_acc': acc,
                'val_acc': val_acc
            }
            with open(f'{HISTORY_DIR}/{targ}_fold{fold_}_history.json', 'w') as f:
                json.dump(history_dict, f, indent=2, ensure_ascii=False)


        weights = model.state_dict()
        torch.save(weights,
                   f'{WEIGHTS_DIR}/model_{targ}_fold{fold_}_epochs{N_EPOCHS}_bs{BS}.pth')


        val_loader = DataLoader(dataset=val_dataset, shuffle=False,
                                batch_size=BS, num_workers=8)
        val_preds = make_val_preds(model, val_loader)
        oof_df.loc[val_idx, targ] = val_preds['predicted']

        print('==================================')
        print(f'{targ} val score on fold {fold_+1}: ')
        print(score(val_preds['labels'], val_preds['predicted']))

        print(f'{targ} val MAE on fold {fold_+1}: ')
        print(mean_absolute_error(val_preds['labels'], val_preds['predicted']))
        print('==================================')

    return None




if __name__ == '__main__':
    oof = kfold_separate_targets()
    overall_score = w_nae_score(train_data[TARGETS].values, oof.values)

    print('==================================')
    print('Training completed.')
    print(f'Overall score : {overall_score}')
    print('==================================')

