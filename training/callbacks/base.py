"""
 File name   : base.py
 Description : callbacks interface

 Date created : 07.03.2021
 Author:  Pytorch Lighting code modified by Ihar Khakholka
"""



class Callback(object):
    r"""
    Abstract base class used to build new callbacks.
    Subclass this class and override any of the relevant hooks
    """

    def setup(self, state, stage: str):
        """Called when fit or test begins"""
        pass

    def teardown(self, state, stage: str):
        """Called when fit or test ends"""
        pass

    def on_init_start(self, state):
        """Called when the state initialization begins, model has not yet been set."""
        pass

    def on_init_end(self, state):
        """Called when the state initialization ends, model has not yet been set."""
        pass

    def on_fit_start(self, state):
        """Called when fit begins"""
        pass

    def on_fit_end(self, state):
        """Called when fit ends"""
        pass

    def on_sanity_check_start(self, state):
        """Called when the validation sanity check starts."""
        pass

    def on_sanity_check_end(self, state):
        """Called when the validation sanity check ends."""
        pass

    def on_train_batch_start(self, state, batch, batch_idx, dataloader_idx):
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(self, state, outputs, batch, batch_idx, dataloader_idx):
        """Called when the train batch ends."""
        pass

    def on_train_epoch_start(self, state):
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, state):
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_start(self, state):
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, state):
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, state):
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, state):
        """Called when the test epoch ends."""
        pass

    def on_epoch_start(self, state):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, state):
        """Called when the epoch ends."""
        pass

    def on_batch_start(self, state):
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(self, state, batch, batch_idx, dataloader_idx):
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(self, state, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(self, state, batch, batch_idx, dataloader_idx):
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(self, state, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        pass

    def on_batch_end(self, state):
        """Called when the training batch ends."""
        pass

    def on_train_start(self, state):
        """Called when the train begins."""
        pass

    def on_train_end(self, state):
        """Called when the train ends."""
        pass

    def on_pretrain_routine_start(self, state):
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, state):
        """Called when the pretrain routine ends."""
        pass

    def on_validation_start(self, state):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, state):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, state):
        """Called when the test begins."""
        pass

    def on_test_end(self, state):
        """Called when the test ends."""
        pass

    def on_keyboard_interrupt(self, state):
        """Called when the training is interrupted by KeyboardInterrupt."""
        pass

    def on_save_checkpoint(self, state):
        """Called when saving a model checkpoint, use to persist state."""
        pass

    def on_load_checkpoint(self, checkpointed_state):
        """Called when loading a model checkpoint, use to reload state."""
        pass

    def on_after_backward(self, state):
        """
        Called after loss.backward() and before optimizers do anything.
        """
        pass

    def on_before_zero_grad(self, state, optimizer):
        """
        Called after optimizer.step() and before optimizer.zero_grad().
        """
        pass
