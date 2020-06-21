import argparse
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
from utils import seed_everything, w_nae_score, score
from utils import fit_epoch, eval_epoch, train, make_val_preds
from losses import loss_function
from optimizers import optimizer


def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch-size', type=int, default=16)
    arg('--lr', type=float, default=0.001)
    arg('--momentum', type=float, default=0.9)
    arg('--weight-decay', type=float, default=0.0001)
    arg('--n-epochs', type=int, default=15)
    arg('--fold', type=int, default=5)
    arg('--n-classes', type=int, default=5)
    arg('--history', type=bool, default=False)
    arg('--seed', type=int, default=111)
    arg('--workers', type=int, default=6)
    arg('--use-targets', type=str, default='all', choices=['all', 'separate'])
    arg('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
    arg('--loss', type=str, default='mae', choices=['mae', 'w_nae'])
    args = parser.parse_args()


    MRI_TR = 'data/fMRI_train/'
    MRI_TE = 'data/fMRI_test/'
    WEIGHTS_DIR = 'model_weights'
    HISTORY_DIR = 'train_history'

    if not os.path.isdir(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)
    if not os.path.isdir(HISTORY_DIR):
        os.mkdir(HISTORY_DIR)

    LR = args.lr
    BS = args.batch_size
    N_EPOCHS = args.n_epochs
    N_FOLDS = args.fold
    N_CLASSES = args.n_classes
    SEED = args.seed
    SAVE_HISTORY = args.history
    USE_TARGETS = args.use_targets
    WORKERS = args.workers
    seed_everything(SEED)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    train_data = pd.read_csv('data/train_scores.csv').dropna().iloc[:5]
    train_data.index = np.arange(train_data.shape[0])
    TARGETS = list(train_data.columns[1:])
    train_data = train_data.iloc[
        train_data.drop('Id', axis=1).drop_duplicates().index.values]
    train_data.index = np.arange(train_data.shape[0])

    def train_kfold_all(save_history=False):
        '''
        This function performs kfold models training
        on all targets together and saves results.
        Returns data frame of predicted validation data.
        :param save_history: bool, save history.json or not.
        :return: oof_df
        '''
        oof_df = pd.DataFrame(data=np.zeros(train_data[TARGETS].shape), columns=TARGETS)
        print('Training on all targets starts.')
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data, train_data)):
            print(f'Fold {fold_ + 1}')

            tr_data = train_data.loc[trn_idx]
            val_data = train_data.loc[val_idx]

            tr_dataset = MRIDataset(MRI_TR, tr_data, 'train', (TARGETS))
            val_dataset = MRIDataset(MRI_TR, val_data, 'val', TARGETS)

            model = create_nn(N_CLASSES)
            opt = optimizer(args.optimizer, model.parameters(), lr=LR,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
            criterion = loss_function(args.loss)

            with torch.autograd.set_detect_anomaly(True):
                history = train(tr_dataset, val_dataset, model, epochs=N_EPOCHS,
                                batch_size=BS, opt=opt, criterion=criterion,
                                workers=WORKERS)

            if save_history:
                loss, acc, val_loss, val_acc = zip(*history)
                history_dict = {
                    'train_loss': loss,
                    'val_loss': val_loss,
                    'train_acc': acc,
                    'val_acc': val_acc
                }
                with open(f'{HISTORY_DIR}/all_fold{fold_}_history.json', 'w') as f:
                    json.dump(history_dict, f, indent=2, ensure_ascii=False)

            weights = model.state_dict()
            torch.save(weights,
                       f'{WEIGHTS_DIR}/model_all_fold{fold_}_epochs{N_EPOCHS}_bs{BS}.pth')

            val_loader = DataLoader(dataset=val_dataset, shuffle=False,
                                    batch_size=BS, num_workers=WORKERS)
            val_preds = make_val_preds(model, val_loader)
            oof_df.loc[val_idx] = val_preds['predicted']

            print('==================================')
            print(f'All val score on fold {fold_ + 1}: ')
            print(w_nae_score(val_preds['labels'], val_preds['predicted']))

            print(f'All val MAE on fold {fold_ + 1}: ')
            print(mean_absolute_error(val_preds['labels'], val_preds['predicted']))
            print('==================================')

        return oof_df

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
            print(f'Fold {fold_ + 1}')

            tr_data = train_data.loc[trn_idx]
            val_data = train_data.loc[val_idx]

            tr_dataset = MRIDataset(MRI_TR, tr_data, 'train', targ)
            val_dataset = MRIDataset(MRI_TR, val_data, 'val', targ)

            model = create_nn(N_CLASSES)
            opt = optimizer(args.optimizer, model.parameters(), lr=LR,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
            criterion = loss_function(args.loss)

            with torch.autograd.set_detect_anomaly(True):
                history = train(tr_dataset, val_dataset, model, epochs=N_EPOCHS,
                                batch_size=BS, opt=opt, criterion=criterion,
                                workers=WORKERS)

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
                                    batch_size=BS, num_workers=WORKERS)
            val_preds = make_val_preds(model, val_loader)
            oof_df.loc[val_idx, targ] = val_preds['predicted']

            print('==================================')
            print(f'{targ} val score on fold {fold_ + 1}: ')
            print(score(val_preds['labels'], val_preds['predicted']))

            print(f'{targ} val MAE on fold {fold_ + 1}: ')
            print(mean_absolute_error(val_preds['labels'], val_preds['predicted']))
            print('==================================')

        return None

    if USE_TARGETS == 'all':
        oof = train_kfold_all(SAVE_HISTORY)
    else:
        kfold_separate_targets()
    overall_score = w_nae_score(train_data[TARGETS].values, oof.values)
    print('==================================')
    print('Training completed.')
    print(f'Overall score : {overall_score}')
    print('==================================')


if __name__ == '__main__':
    main()


