"""
 File name   : callbacks_container.py
 Description : Class to manage callbacks

 Date created : 07.03.2021
 Author:  Ihar Khakholka
"""


import datetime
from addict import Dict

def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")



class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.state = Dict({})

    def append(self, callback):
        self.callbacks.append(callback)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin( self.state)

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self.state)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_batch_end(batch)

    def on_train_start(self):
        self.state.start_time = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_start(self.state)

    def on_train_end(self):
        self.state.stop_time = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(self.state)

    def on_validation_end(self):
        for callback in self.callbacks:
            callback.on_validation_end(self.state)
