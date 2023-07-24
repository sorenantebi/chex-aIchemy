
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import torchxrayvision as xrv
from skimage.io import imread
from tqdm import tqdm

""" 
Todo

"""
class CheXpertDataset(Dataset):
    def __init__(self, img_data_dir, csv_file_img, image_size, model_name, augmentation=False, pseudo_rgb = True, label_noise = False):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.img_data_dir = img_data_dir
        self.model_name = model_name

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']


        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = self.img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label_disease = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label_disease[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            img_label_sex = np.array(self.data.loc[idx, 'sex_label'], dtype='int64')
            img_label_race = np.array(self.data.loc[idx, 'race_label'], dtype='int64')

            if label_noise:
                noise_p = 0.04
                add_noise = np.random.binomial(n=2, p=noise_p) == 1

                # UNDERDIAGNOSIS
                if img_label_sex == 1 and img_label_disease[0] == 0 and add_noise:
                # if img_label_race == 2 and img_label_disease[0] == 0 and add_noise:
                    img_label_disease[0] = 1
                    for i in range(1, len(self.labels)):
                        img_label_disease[i] = 0

                # RANDOM NOISE
                # if img_label_sex == 1 and add_noise:
                # if img_label_race == 2 and add_noise:
                #     for i in range(0, len(self.labels)):
                #         img_label_disease[i] = np.array(np.random.choice(2), dtype='float32')

            sample = {'image_path': img_path, 'label_disease': img_label_disease, 'label_sex': img_label_sex, 'label_race': img_label_race}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label_disease = torch.from_numpy(sample['label_disease'])
        label_sex = torch.from_numpy(sample['label_sex'])
        label_race = torch.from_numpy(sample['label_race'])

        if self.model_name == 'imagenet':
            if self.pseudo_rgb:
                image = image.repeat(3, 1, 1)

        if self.do_augment:
            image = self.augment(image)

        return {'image': image, 'label_disease': label_disease, 'label_sex': label_sex, 'label_race': label_race}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        if self.model_name != 'imagenet':
            image = xrv.datasets.normalize(image, 255)
        return {'image': image, 'label_disease': sample['label_disease'], 'label_sex': sample['label_sex'], 'label_race': sample['label_race']}

"""
/Todo 

"""
class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, img_data_dir, csv_train_img, csv_val_img, csv_test_img, image_size, pseudo_rgb, batch_size, num_workers, model_name):
        super().__init__()
        self.img_data_dir = img_data_dir
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.img_data_dir, self.csv_train_img, self.image_size, model_name=model_name, augmentation=True, pseudo_rgb=pseudo_rgb, label_noise=False)
        self.val_set = CheXpertDataset(self.img_data_dir, self.csv_val_img, self.image_size, model_name=model_name, augmentation=False, pseudo_rgb=pseudo_rgb, label_noise=False)
        self.test_set = CheXpertDataset(self.img_data_dir, self.csv_test_img, self.image_size, model_name=model_name, augmentation=False, pseudo_rgb=pseudo_rgb, label_noise=False)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

