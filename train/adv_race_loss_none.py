import os
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
import torchxrayvision as xrv
from checkpoint import CustomModelCheckPoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

model_name = 'pc'
image_size = (224, 224)
num_classes_disease = 14
num_classes_sex = 2
num_classes_race = 3
class_weights_race = (1.0, 1.0, 1.0) # helps with balancing accuracy, very little impact on AUC
batch_size = 150
epochs = 5
alpha = 0.0001
num_workers = 4
fading_in_steps = 5000
fading_in_range = 200
# img_data_dir = '<path_to_data>/CheXpert-v1.0/'
# img_data_dir = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
img_data_dir = '/rds/general/user/sea22/ephemeral/datafiles/chexpert/'


class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, augmentation=False, pseudo_rgb = True, label_noise = False):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

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
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
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

        """ if self.pseudo_rgb:
            image = image.repeat(3, 1, 1) """

        if self.do_augment:
            image = self.augment(image)

        return {'image': image, 'label_disease': label_disease, 'label_sex': label_sex, 'label_race': label_race}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)
        image = xrv.datasets.normalize(image, 255)
        return {'image': image, 'label_disease': sample['label_disease'], 'label_sex': sample['label_sex'], 'label_race': sample['label_race']}


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, image_size, pseudo_rgb, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, augmentation=True, pseudo_rgb=pseudo_rgb, label_noise=False)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb, label_noise=False)
        self.test_set = CheXpertDataset(self.csv_test_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb, label_noise=False)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race, alpha):
        super().__init__()
        self.num_classes_disease = num_classes_disease
        self.num_classes_sex = num_classes_sex
        self.num_classes_race = num_classes_race
        self.alpha = alpha
        self.class_weights_race = torch.FloatTensor(class_weights_race)
        self.backbone = xrv.models.DenseNet(weights=f"densenet121-res224-{model_name}")
        self.backbone.op_threshs = None
        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect

    def forward(self, x):
        embedding = self.backbone.forward(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        params_sex = list(self.fc_sex.parameters())
        params_race =  list(self.fc_race.parameters())
        optim_backbone = torch.optim.Adam(params_backbone, lr=0.001)
        optim_disease = torch.optim.Adam(params_disease, lr=0.001)
        optim_sex = torch.optim.Adam(params_sex, lr=0.001)
        optim_race = torch.optim.Adam(params_race, lr=0.001)
        return optim_disease, optim_sex, optim_race, optim_backbone

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        out_disease, out_sex, out_race = self.forward(img)
        loss_disease = F.binary_cross_entropy(torch.sigmoid(out_disease), lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))
        loss_confusion = -torch.mean(torch.log_softmax(out_race, dim=1))
        # loss_confusion = -torch.mean(torch.log_softmax(out_sex, dim=1))
        return loss_disease, loss_sex, loss_race, loss_confusion

    def training_step(self, batch, batch_idx, optimizer_idx):
        omega = self.alpha / (1 + np.exp(-(self.global_step - fading_in_steps) / fading_in_range))
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race, "train_loss_confusion": loss_confusion, "omega": omega})
        # grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        # self.logger.experiment.add_image('images', grid, self.global_step)
        
        if optimizer_idx == 0:
            return loss_disease #+ omega * loss_confusion
        if optimizer_idx == 1:

            return loss_sex
        if optimizer_idx == 2:
            return loss_race
        if optimizer_idx == 3:
            return - omega * loss_race # V1: negate race classification lose
            #return - omega * loss_sex  # V1: negate sex classification lose
            # return omega * loss_confusion # V2: confusion loss

    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        loss_disease_mod = loss_disease if self.global_step > fading_in_steps else 1
        self.log_dict({"val_loss_disease_mod": loss_disease_mod, "val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race, "val_loss_confusion": loss_confusion})

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race, "test_loss_confusion": loss_confusion})


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def test(model, data_loader, device):
    model.eval()
    logits_disease = []
    preds_disease = []
    targets_disease = []
    logits_sex = []
    preds_sex = []
    targets_sex = []
    logits_race = []
    preds_race = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            out_disease, out_sex, out_race = model(img)

            pred_disease = torch.sigmoid(out_disease)
            pred_sex = torch.softmax(out_sex, dim=1)
            pred_race = torch.softmax(out_race, dim=1)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            logits_sex.append(out_sex)
            preds_sex.append(pred_sex)
            targets_sex.append(lab_sex)

            logits_race.append(out_race)
            preds_race.append(pred_race)
            targets_race.append(lab_race)

        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)

        logits_sex = torch.cat(logits_sex, dim=0)
        preds_sex = torch.cat(preds_sex, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)

        logits_race = torch.cat(logits_race, dim=0)
        preds_race = torch.cat(preds_race, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

        counts = []
        for i in range(0,num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_sex):
            t = targets_sex == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_race):
            t = targets_race == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds_disease.cpu().numpy(), targets_disease.cpu().numpy(), logits_disease.cpu().numpy(), preds_sex.cpu().numpy(), targets_sex.cpu().numpy(), logits_sex.cpu().numpy(), preds_race.cpu().numpy(), targets_race.cpu().numpy(), logits_race.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets_disease = []
    targets_sex = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            emb = model.backbone(img)
            embeds.append(emb)
            targets_disease.append(lab_disease)
            targets_sex.append(lab_sex)
            targets_race.append(lab_race)

        embeds = torch.cat(embeds, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

    return embeds.cpu().numpy(), targets_disease.cpu().numpy(), targets_sex.cpu().numpy(), targets_race.cpu().numpy()


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)
    # pl.seed_everything(14, workers=True)
    #pl.seed_everything(96, workers=True)

    # data
    data = CheXpertDataModule(csv_train_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.train.csv',
                              csv_val_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.val.csv',
                              csv_test_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.resample.test.csv',
                              image_size=image_size,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # model
    model_type = DenseNet
    model = model_type(num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race, alpha=alpha)

    # Create output directory
    out_name = f'densenet-{model_name}-customcheck'
    out_dir = 'adversarial/customcheck/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    checkpoint_callback = CustomModelCheckPoint(fading_in_steps=fading_in_steps, monitor="val_loss_disease_mod", mode='min')
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        logger=TensorBoardLogger('adversarial/customcheck/', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)
