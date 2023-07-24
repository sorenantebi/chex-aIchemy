import os
import torch
import sys
sys.path.insert(1, '/rds/general/user/sea22/home/PROJECT/misc/')


import numpy as np

import torchvision.transforms as T

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
from datasets.CheXDataModule import CheXpertDataModule
from models.models import DenseNetXRVAdversarial
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
    data = CheXpertDataModule(img_data_dir= args.img_data_dir,
                              csv_train_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.train.csv',
                              csv_val_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.sample.val.csv',
                              csv_test_img='/rds/general/user/sea22/ephemeral/datafiles/chexpert/CheXpert-v1.0/chexpert.resample.test.csv',
                              image_size=tuple(args.image_size),
                              pseudo_rgb=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              model_name=args.model_name)

    # model
    """
    confusion = 'sex-negation', 'race-negation', 'sex-confusion', 'race-confusion', None
    """
    model_type = DenseNetXRVAdversarial
    model = model_type(args=args)

    # Create output directory
    out_name = f'densenet-{args.model_name}-{args.confusion}-{args.alpha}'
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

    if args.confusion is None:
        checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", mode='min')
    else: 
        checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease_mod", mode='min')

    #todo hyperparam tuning
    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=TensorBoardLogger('results/', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)

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
