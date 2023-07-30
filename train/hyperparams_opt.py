import sys
import optuna
import logging
import matplotlib.pylab as plt
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from argparse import ArgumentParser
from misc.datasets.CheXDataModule import CheXpertDataModule
from misc.models.models import DenseNetMultitask


"""
Learning rate tuning using Optuna Search algorithm
"""

def objective(args, trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    args.lr_d = trial.suggest_loguniform("lr_d", 1e-4, 1e-1)
    args.lr_s = trial.suggest_loguniform("lr_s", 1e-4, 1e-1)
    args.lr_r = trial.suggest_loguniform("lr_r", 1e-4, 1e-1)
    args.lr_b = trial.suggest_loguniform("lr_b", 1e-4, 1e-1)

    model = DenseNetMultitask(args=args)
    datamodule = CheXpertDataModule(args=args,
                              csv_train_img='/vol/biomedic3/bglocker/msc2023/sea22/datafiles/chexpert/CheXpert-v1.0/c.train.csv',
                              csv_val_img='/vol/biomedic3/bglocker/msc2023/sea22/datafiles/chexpert/CheXpert-v1.0/c.val.csv',
                              csv_test_img='/vol/biomedic3/bglocker/msc2023/sea22/datafiles/chexpert/CheXpert-v1.0/c.test.csv',
                              pseudo_rgb=True
                              )

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss_disease")],
    )
    hyperparameters = dict(lr_d=args.lr_d, lr_s=args.lr_s, lr_r=args.lr_r, lr_b=args.lr_b)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss_disease"].item()


if __name__ == "__main__":
    from params import setup_hparams, add_arguments
 
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)

    pruner =  optuna.pruners.MedianPruner() #optuna.pruners.NopPruner()
    """ optuna.pruners.MedianPruner() if args.pruning else """
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction="minimize", pruner=pruner, study_name=study_name, storage=storage_name)
    
    study.optimize(lambda trial: objective(args=args, trial=trial), n_trials=30, gc_after_trial=True)
    optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.savefig("Optimization history")
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

