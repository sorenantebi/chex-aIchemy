import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.io import imsave
from argparse import ArgumentParser
from misc.datasets.CheXDataModule import CheXpertDataModule
from misc.models.models import DenseNetMultitask
from utils import analysis
from checkpoint import CustomModelCheckPoint
import json

def main(args):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)
    

    # data
    data = CheXpertDataModule(args=args,
                              csv_train_img='/<path-to-train.csv>',
                              csv_val_img='/<path-to-val.csv>',
                              csv_test_img='/<path-to-test.csv>',
                              pseudo_rgb=True)

    # model
    """
    Confusion type --> confusion = 'sex-negation' / 'race-negation' / 'sex-confusion' / 'race-confusion' / None
    """
    model_type = DenseNetMultitask # or DenseNet
    model = model_type(args=args)

    # Create output directory
    out_name = f'densenet-{args.model_name}-{args.confusion}-{args.alpha}-{args.label_noise}' # edit as needed
    out_dir = f'../../results/{out_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    hparam_config = os.path.join(out_dir, 'hparam_config')
    if not os.path.exists(hparam_config):
        os.makedirs(hparam_config)
    
    with open(os.path.join(hparam_config, 'hparams.json'), "w") as json_file:
        json_file.write(json.dumps(args.__dict__))
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(
            os.path.join(temp_dir, f'sample_{str(idx)}.jpg'),
            sample['image'].astype(np.uint8),
        )

    monitor = "val_loss_disease" if args.confusion is None else "val_loss_disease_mod"
    checkpoint_callback = CustomModelCheckPoint(fading_in_steps=args.fading_in_steps, monitor=monitor, mode='min')

    
    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=args.epochs,
        accelerator = "gpu",
        devices = args.gpus,
        logger=TensorBoardLogger('../../results/', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{str(args.dev)}" if use_cuda else "cpu")

    model.to(device)

    # configure according to single vs multitask learning
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
