"""
 File name   : callbacks_factory.py
 Description : Callbacks factory

 Date created : 07.03.2021
 Author:  Ihar Khakholka
"""


import importlib

class CallbacksFactory:

    @staticmethod
    def parse_callbacks(callbacks: dict):
        callbacks_module = importlib.import_module('training.callbacks')

        callbacks_list = []
        for c_name, params in callbacks.items():
            c_name = c_name.split('__')[0]

            callback_class_ = getattr(callbacks_module, c_name)
            callback = callback_class_(**params)
            callbacks_list.append(callback)

        callbacks_list = callbacks_list[::-1]
        return callbacks_list
