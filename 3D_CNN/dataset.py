import torch
from torch.utils.data import Dataset
import h5py


class MRIDataset(Dataset):
    '''
    Created by Dmitriy Ershov.

    Pytorch Dataset for MRI data
    in TReNDS kaggle competition.

    Args:

    ================================================

    path: str
        path to files
    df: pandas.DataFrame
        dataframe with column called 'Id' (and target, if not test)
    mode: str
        one of [train, val, test]
    target: str or list
        target's column name for train and validation
    '''

    def __init__(self, path, df, mode, target=None):
        super(MRIDataset, self).__init__()

        self.files = df['Id'].values
        self.mode = mode
        self.path = path

        TARGETS = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
        if target:
            self.target = target
            if isinstance(self.target, list):
                for targ in self.target:
                    if targ not in TARGETS:
                        print(f"{targ} is not correct; correct modes: {TARGETS}")
                        raise NameError
            elif self.target not in TARGETS:
                print(f"{self.target} is not correct; correct modes: {TARGETS}")
                raise NameError

        DATA_MODES = ('train', 'val', 'test')
        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        if self.mode != 'test':
            self.labels = df[self.target].values

        self.len_ = len(self.files)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        file = self.path + str(file) + '.mat'
        with h5py.File(file) as h5_data:
            image = h5_data['SM_feature'][()]
        return image

    def _prepare_sample(self, image):
        image = torch.FloatTensor(image)
        return image

    def __getitem__(self, index):

        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)

        if self.mode == 'test':
            return x
        else:
            y = self.labels[index]
            return x, y