from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

"""
Only save checkpoints if global_step > fading_in_steps 
"""
class CustomModelCheckPoint(ModelCheckpoint):
    def __init__(self, fading_in_steps, *args, **kwargs):
         super().__init__(*args, **kwargs)
         self.fading_in_steps = fading_in_steps
        
    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
            trainer.global_step < self.fading_in_steps
            or bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )