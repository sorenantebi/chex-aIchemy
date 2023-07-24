import os
import torch
import sys
sys.path.insert(1, '/rds/general/user/sea22/home/PROJECT/misc/')
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

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
from datasets.CheXDataModule import CheXpertDataModule
from models.models import DenseNetXRVAdversarial
from utils import test_multi, embeddings
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


model_name = 'pc'
image_size = (224, 224)
num_classes_disease = 14
num_classes_sex = 2
num_classes_race = 3
class_weights_race = (1.0, 1.0, 1.0) # helps with balancing accuracy, very little impact on AUC
batch_size = 150
epochs = 40
num_workers = 4
fading_in_steps = 5000
fading_in_range = 200

img_data_dir = '/rds/general/user/sea22/ephemeral/datafiles/chexpert/'


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)
    # pl.seed_everything(14, workers=True)
    #pl.seed_everything(96, workers=True)
    confusion = str(args.confusion) 
    print(f"CONFUSION TYPE: {confusion}")
    # 4- 6
    alpha = 1*10**-(int(args.alpha))
    print(f"ALPHA: {alpha}")

    # data
    data = CheXpertDataModule(img_data_dir= img_data_dir,
                              csv_train_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.train.csv',
                              csv_val_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.val.csv',
                              csv_test_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.resample.test.csv',
                              image_size=image_size,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # model
    """
    confusion = 'sex-negation', 'race-negation', 'sex-confusion', 'race-confusion'
    """
    model_type = DenseNetXRVAdversarial
    model = model_type(model_name=model_name,
                       num_classes_disease=num_classes_disease, 
                       num_classes_sex=num_classes_sex, 
                       num_classes_race=num_classes_race, 
                       class_weights_race=class_weights_race, 
                       alpha=alpha,
                       fading_in_steps=fading_in_steps,
                       fading_in_range=fading_in_range,
                       confusion= confusion)

    # Create output directory
    out_name = f'densenet-{model_name}-{confusion}-{alpha}'
    out_dir = 'adversarial/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease_mod", mode='min')
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger('adversarial/', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race, alpha=alpha)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")
    
    model.to(device)

    cols_names_classes_disease = ['class_' + str(i) for i in range(0,num_classes_disease)]
    cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
    cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

    cols_names_classes_sex = ['class_' + str(i) for i in range(0,num_classes_sex)]
    cols_names_logits_sex = ['logit_' + str(i) for i in range(0, num_classes_sex)]

    cols_names_classes_race = ['class_' + str(i) for i in range(0,num_classes_race)]
    cols_names_logits_race = ['logit_' + str(i) for i in range(0, num_classes_race)]

    print('VALIDATION')
    preds_val_disease, targets_val_disease, logits_val_disease, preds_val_sex, targets_val_sex, logits_val_sex, preds_val_race, targets_val_race, logits_val_race = test_multi(model, data.val_dataloader(), device)
    
    df = pd.DataFrame(data=preds_val_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_val_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions_val_disease.csv'), index=False)

    df = pd.DataFrame(data=preds_val_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_val_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_sex
    df.to_csv(os.path.join(out_dir, 'predictions_val_sex.csv'), index=False)

    df = pd.DataFrame(data=preds_val_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_val_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'predictions_val_race.csv'), index=False)

    print('TESTING')
    preds_test_disease, targets_test_disease, logits_test_disease, preds_test_sex, targets_test_sex, logits_test_sex, preds_test_race, targets_test_race, logits_test_race = test_multi(model, data.test_dataloader(), device)
    
    df = pd.DataFrame(data=preds_test_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_test_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions_test_disease.csv'), index=False)

    df = pd.DataFrame(data=preds_test_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_test_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_sex
    df.to_csv(os.path.join(out_dir, 'predictions_test_sex.csv'), index=False)

    df = pd.DataFrame(data=preds_test_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_test_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'predictions_test_race.csv'), index=False)

    print('EMBEDDINGS')

    embeds_val, targets_val_disease, targets_val_sex, targets_val_race = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets_disease = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_val_sex
    df['target_race'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'embeddings_val.csv'), index=False)

    embeds_test, targets_test_disease, targets_test_sex, targets_test_race = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets_disease = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_test_sex
    df['target_race'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'embeddings_test.csv'), index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--confusion', required=True)
    parser.add_argument('--alpha', required=True)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)
