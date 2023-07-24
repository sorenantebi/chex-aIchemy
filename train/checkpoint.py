from pytorch_lightning.callbacks import ModelCheckpoint


class CustomModelCheckPoint(ModelCheckpoint):
    def __init__(self, fading_in_steps, *args, **kwargs):
         super().__init__(*args, **kwargs)
         self.fading_in_steps = fading_in_steps

    def save_checkpoint(self, trainer, pl_module):
            """
            Performs the main logic around saving a checkpoint.
            This method runs on all ranks, it is the responsibility of `self.save_function`
            to handle correct behaviour in distributed training, i.e., saving only on rank 0.
            """
            epoch = trainer.current_epoch
            global_step = trainer.global_step

            if (
                global_step < self.fading_in_steps
                or self.save_top_k == 0  # no models are saved
                or self.period < 1  # no models are saved
                or (epoch + 1) % self.period  # skip epoch
                or trainer.running_sanity_check  # don't save anything during sanity check
                or self.last_global_step_saved == global_step  # already saved at the last step
            ):
                return

            self._add_backward_monitor_support(trainer)
            self._validate_monitor_key(trainer)

            # track epoch when ckpt was last checked
            self.last_global_step_saved = global_step

            # what can be monitored
            monitor_candidates = self._monitor_candidates(trainer)

            # ie: path/val_loss=0.5.ckpt
            filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, epoch, global_step)

            # callback supports multiple simultaneous modes
            # here we call each mode sequentially
            # Mode 1: save all checkpoints OR only the top k
            if self.save_top_k:
                self._save_top_k_checkpoints(monitor_candidates, trainer, pl_module, filepath)

            # Mode 2: save the last checkpoint
            self._save_last_checkpoint(trainer, pl_module, monitor_candidates, filepath)