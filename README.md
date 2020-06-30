# TReNDS-Neuroimaging 
* https://www.kaggle.com/c/trends-assessment-prediction/


This is a pytorch code for 3D CNN in TReNDS Neuroimaging kaggle competition \
Author - https://www.kaggle.com/bejeweled \
Pytorch lightning version (used for training) written by my teammate 
https://www.kaggle.com/tiulpin and will be available on his github https://github.com/tiulpin



# Available arguments :

* Batch size   --batch-size, type=int, default=16
* Learning rate   --lr, type=float, default=0.001
* Momentum   --momentum, type=float, default=0.9
* Optimizer weight decay   --weight-decay, type=float, default=0.0001
* Number of epochs   --n-epochs, type=int, default=15
* Number of folds for training   --fold, type=int, default=5
* Number of classes for NN   --n-classes, type=int, default=5
* Save json history   --history, type=bool, default=False
* Seed for seeding everything   --seed, type=int, default=111
* Number of workers in dataloader   --workers, type=int, default=6
* Train on 5 classes or on 1   --use-targets, type=str, default='all', choices=['all', 'separate']
* Optimizer for NN   --optimizer, type=str, default='sgd', choices=['sgd', 'adam', 'adamw']
* Criterion (loss function)   --loss, type=str, default='mae', choices=['mae', 'w_nae']



# Base directories names for data and saving weights & history

MRI_TR = 'data/fMRI_train/' \
MRI_TE = 'data/fMRI_test/' \
WEIGHTS_DIR = 'model_weights' \
HISTORY_DIR = 'train_history' \

You can change it in 3D_CNN/main.py



# Some details

For training in PL version data was converted to numpy for faster data reading
We trained with **--n-classes** 5 and **W_NAE loss**, **SGD** gives better results, 
**parameters** was default. \
Also, in PL version we used **early stopping** and **lr scheduler**



# Non - 3D CNN Models

Final submission is a blend of different ensembles and models \
Full list of models : \
**SVR** \
**Ridge** \
**Bayesian Ridge** \
**Lasso** \
**ElasticNet** \
**Bagging Regressor** \
**LGBM** (mix of DART and GBDT) \
**DNN** for tabular data



# Final

* All of it was used in many ensembles, but also as solo models with different preprocessing, normalizations and etc.
* Final **W_NAE score** is **.15820** and **27 place** in public LB and **.15853** and **31 place** in private LB