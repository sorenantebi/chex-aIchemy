import os
import torch
import sys



import numpy as np

import torchvision.transforms as T

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
from misc.datasets.CheXDataModule import CheXpertDataModule
from misc.models.models import DenseNetMultitask
from utils import analysis


import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def main(args):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)
    # pl.seed_everything(14, workers=True)
    #pl.seed_everything(96, workers=True)

    # data
    data = CheXpertDataModule(args=args,
                              csv_train_img='/vol/biomedic3/bglocker/msc2023/sea22/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.train.csv',
                              csv_val_img='/vol/biomedic3/bglocker/msc2023/sea22/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.val.csv',
                              csv_test_img='/vol/biomedic3/bglocker/msc2023/sea22/datafiles/chexpert/CheXpert-v1.0/chexpert.resample.test.csv',
                              pseudo_rgb=True)

    # model
    """
    confusion = 'sex-negation', 'race-negation', 'sex-confusion', 'race-confusion', None
    """
    model_type = DenseNetMultitask
    model = model_type(args=args)

    # Create output directory
    out_name = f'densenet-{args.model_name}-{args.confusion}-{args.alpha}-test'
    out_dir = f'results/{out_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(
            os.path.join(temp_dir, f'sample_{str(idx)}.jpg'),
            sample['image'].astype(np.uint8),
        )

    """  monitor = "val_loss_disease" if args.confusion is None else "val_loss_disease_mod"
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='min')

    #todo hyperparam tuning
    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=args.epochs,
        accelerator = "gpu",
        devices = args.gpus,
        logger=TensorBoardLogger('results/', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data) """

    model = model_type.load_from_checkpoint('/homes/sea22/MSC_PROJECT/main/chex-aIchemy/train/results/densenet-imagenet-None-0.0001/version_0/checkpoints/epoch=9-step=15270.ckpt', args=args)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{str(args.dev)}" if use_cuda else "cpu")

    model.to(device)
    analysis(out_dir=out_dir,
             num_classes_disease=args.num_classes_disease, 
             num_classes_sex=args.num_classes_sex, 
             num_classes_race=args.num_classes_race,
             model=model,
             data=data,
             device=device)

if __name__ == '__main__':
    from params import setup_hparams, add_arguments
    
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)

    main(args)
