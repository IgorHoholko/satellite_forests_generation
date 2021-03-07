"""
 File name   : early_stopping.py
 Description : Early stopping callback

 Date created : 07.03.2021
 Author:  Ihar Khakholka
"""


import numpy as np
import logging

from training.callbacks.base import Callback



class EarlyStopping(Callback):
    r"""
    Monitor a validation metric and stop training when it stops improving.
    Args:
        monitor: quantity to be monitored. Default: ``'early_stop_on'``.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0.0``.
        patience: number of validation epochs with no improvement
            after which training will be stopped. Default: ``3``.
        verbose: verbosity mode. Default: ``False``.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: ``'auto'``.
        strict: whether to crash the training if `monitor` is
            not found in the validation metrics. Default: ``True``.

    """
    mode_dict = {
        'min': np.less,
        'max': np.greater,
    }

    def __init__(self, monitor: str = '', min_delta: float = 0.0, patience: int = 3,
                 verbose: bool = False, mode: str = 'auto', strict: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta

        self.wait_count = 0
        self.stopped_epoch = 0
        self.mode = mode
        self.warned_result_obj = False
        # Indicates, if eval results are used as basis for early stopping
        # It is set to False initially and overwritten, if eval results have been validated

        if mode not in self.mode_dict and mode != 'auto':
            if self.verbose > 0:
                logging.info(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'

        if self.mode == 'auto':
            if self.monitor == 'acc':
                self.mode = 'max'
            else:
                self.mode = 'min'
            if self.verbose > 0:
                logging.info(f'EarlyStopping mode set to {self.mode} for monitoring {self.monitor}.')

        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.inf if self.monitor_op == np.less else -np.inf


    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]

    def on_save_checkpoint(self, state):
        return {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'best_score': self.best_score,
            'patience': self.patience
        }

    def on_load_checkpoint(self, checkpointed_state):
        self.wait_count = checkpointed_state['wait_count']
        self.stopped_epoch = checkpointed_state['stopped_epoch']
        self.best_score = checkpointed_state['best_score']
        self.patience = checkpointed_state['patience']

    def on_validation_end(self, state):
        self._run_early_stopping_check(state)


    def _run_early_stopping_check(self, state):
        """
        Checks whether the early stopping condition is met
        and if so tells the state to stop the training.
        """
        should_stop = False
        current = state.get(self.monitor)

        if not current:
            raise ValueError(f"Can't track {self.monitor}. Check if you log it")

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience

            if bool(should_stop):
                self.stopped_epoch = state.current_epoch
                state.should_stop = True

        # stop every ddp process if any world process decides to stop
        state.should_stop = should_stop
