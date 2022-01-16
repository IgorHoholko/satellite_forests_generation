"""
 File name   : logger.py
 Description : description

 Date created : 12.08.2021
 Author:  Ihar Khakholka
"""


import logging

LEVELS = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

def init_logger(level: str):
    level_ = LEVELS.get(level.lower())
    if level_ is None:
        raise ValueError(
            f"log level given: {level}"
            f" -- must be one of: {' | '.join(LEVELS.keys())}")

    logging.basicConfig(
        level=level_,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
