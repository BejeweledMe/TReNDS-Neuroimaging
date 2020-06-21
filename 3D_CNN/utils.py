import random
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import DataLoader


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def score(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

def w_nae_score(y_true, y_pred):
    w = np.array([0.3, 0.175, 0.175, 0.175, 0.175])
    return np.sum(w*np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


def fit_epoch(model, train_loader, criterion, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += mean_absolute_error(labels.cpu().numpy(), outputs.cpu().detach().numpy())
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects / processed_data
    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += mean_absolute_error(labels.cpu().numpy(), outputs.cpu().numpy())
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects / processed_size
    return val_loss, val_acc


def train(train_set, val_set, model, epochs, batch_size, opt, criterion, workers=6):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=workers)

    history = []
    log_template = '\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}'

    with tqdm(desc='epoch', total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
            print('loss', train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss, \
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

    return history


def make_val_preds(model, val_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    preds = {'predicted': [], 'labels': []}

    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        preds['predicted'].extend(outputs.cpu().numpy())
        preds['labels'].extend(labels.numpy())

    preds['predicted'] = np.array(preds['predicted']).squeeze()
    preds['labels'] = np.array(preds['labels']).squeeze()

    return preds