"""
 File name   : model_checkpoint.py
 Description : class to manage model saving

 Date created : 07.03.2021
 Author:  Ihar Khakholka
"""


from typing import Any, Dict, Optional

import numpy as np
from training.callbacks.base import Callback



class ModelCheckpoint(Callback):
    r"""
    Save the model after every epoch by monitoring a quantity.
    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to retrieve its score.
    """

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"

    mode_dict = {
        'min': np.less,
        'max': np.greater,
    }

    def __init__(
        self,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        save_last: Optional[bool] = None,
        save_top_k: Optional[int] = None,
        save_each: int = None
    ):
        super().__init__()
        self.monitor = monitor
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.mode = mode
        self.save_each = save_each


        self.kth_best_model_path = []
        self.best_model_name = ''
        self.last_model_name = ''
        self.best_model_score = np.inf

        if mode:
            if mode not in self.mode_dict.keys():
                raise ValueError(f'ModelCheckpoint mode {mode} is unknown. Try max ot min mode')
            self.best_model_score *= 1 if mode == 'min' else -1

        if save_top_k is None and monitor is not None:
            self.save_top_k = 1

    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]


    def on_validation_end(self, state):
        """
        checkpoints can be saved at the end of the val loop
        """
        self.save_checkpoint(state)

    def on_save_checkpoint(self, state) -> Dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_name": self.best_model_name,
        }

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        self.best_model_score = checkpointed_state["best_model_score"]
        self.best_model_name = checkpointed_state["best_model_name"]

    def save_checkpoint(self, state):
        """
        Performs the main logic around saving a checkpoint.
        This method runs on all ranks, it is the responsibility of `self.save_function`
        to handle correct behaviour in distributed training, i.e., saving only on rank 0.
        """
        epoch = state.epoch
        if not self.monitor:
            state.model.save_weights()
            return

        model = state.model
        name = model.name
        current_score = state[self.monitor]
        if self.monitor:
            if self.monitor_op(current_score, self.best_model_score):
                new_name = self._form_name(name, self.monitor, postfix='best')
                self.best_model_name = new_name
                self.best_model_score = current_score
                state.model.save_weights(new_name)

        if self.save_last:
            new_name = self._form_name(name, postfix='last')
            self.last_model_name = new_name
            state.model.save_weights(new_name)

        if self.save_each and epoch != 0:
            if epoch % self.save_each == 0:
                new_name = self._form_name(name, epoch = epoch,)
                state.model.save_weights(new_name)

        state.best_model_name = self.best_model_name



    def _form_name(self,
                filename: Optional[str],
                monitor: str = None,
                epoch: int = None,
                score: float = None,
                postfix = ''):

        parts = [filename]
        if monitor: parts.append(monitor)
        if score  : parts.append(score)
        if epoch  : parts.append(f'ep{epoch}')
        if postfix   : parts.append(postfix)

        return self.CHECKPOINT_JOIN_CHAR.join([
            str(part) for part in parts
        ])
