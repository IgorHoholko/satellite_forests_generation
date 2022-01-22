"""
 File name   : train_lit.py
 Description : description

 Date created : 22.01.2022
 Author:  Ihar Khakholka
"""

import logging
import os
from pathlib import Path
import sys
import re
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule, Trainer, seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase
import torch

from training.lit.datamodules import BaseDatamodule
from training.utils import log_hyperparameters, get_requirements
from training.utils.advanced_logging import log_txt, log_model_torch, log_artifact
from forest_generator.utils import load_weights


PROJECT_DIR = Path(__file__).resolve().parents[2]


def train_lit(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    # user_specified_tags = {}  # your own tags here
    # user_specified_tags[MLFLOW_RUN_NAME] = "run name"
    # tags = context_registry.resolve_tags(user_specified_tags)

    if config.get("seed"):
        seed_everything(config.seed)


    regex = r"experiment=([^\s]*)"
    run_command = " ".join(sys.argv)

    experiment = re.search(regex, run_command, re.MULTILINE)
    experiment = experiment.group(1) if experiment else ''
    if '.yaml' in experiment:
        experiment = experiment[:-5]

    # Set run name
    if config.get('logger', None):
        if not config.logger.mlflow.tags:
            config.logger.mlflow.tags = {}
        tags = {}
        if experiment:
            tags.update({'mlflow.runName': experiment})
        if config.parent_run_id:
            tags.update({'mlflow.parentRunId': config.parent_run_id})
        config.logger.mlflow.tags = {**config.logger.mlflow.tags, **tags}

    # Init lightning datamodule
    logging.log(logging.INFO, f"Instantiating datamodule..")
    datamodule: LightningDataModule = BaseDatamodule(config)

    # Init lightning model
    logging.log(logging.INFO, f"Instantiating model <{config.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(config.model, config.model, _recursive_=False)
    logging.log(logging.INFO, f"Instantiating Lit model class for training <{config.model.lit._target_}>")
    model_lit: LightningModule = hydra.utils.instantiate(config.model.lit, model, config, _recursive_=False)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config and config.callbacks:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                logging.log(logging.INFO,f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
            else:
                raise RuntimeError(f"Specifu _base_ for {cb_conf} callback!")

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config and hasattr(config.logger, 'keys'):
        for name, lg_conf in config.logger.items():
            if name == 'none' or name == 'none.yaml':
                continue
            if "_target_" in lg_conf:
                logging.log(logging.INFO, f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
            else:
                raise RuntimeError(f"Specifu _base_ for {lg_conf} logger!")

    # Init lightning trainer
    logging.log(logging.INFO,f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial", deterministic=bool(config.get("seed"))
    )

    # Load weights if checkpoint is provided. THE ONLY WEIGHTS!!!
    if config.get('checkpoint', None):
        logging.log(logging.INFO, f"Loading weights from <{config.checkpoint}>")
        model_lit = load_weights(model_lit, config.checkpoint)


    # Log command and experiment.
    experiment_path = str(PROJECT_DIR / 'configs' / 'experiment' / f"{experiment}.yaml")
    log_artifact(trainer.logger.experiment, trainer.logger.version, experiment_path, "./")
    log_txt(trainer.logger.experiment, trainer.logger.version, run_command, "command.txt")

    # Send some parameters from config to all lightning loggers
    logging.log(logging.INFO,"Logging hyperparameters!")
    log_hyperparameters(config, model_lit, trainer)

    # Train the model
    logging.log(logging.INFO,"Starting training!")
    trainer.fit(model=model_lit, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        logging.log(logging.INFO,"Starting testing!")
        trainer.test()


    # Send currently installed in enviroment packages to all lightning loggers
    logging.log(logging.INFO, "Logging requirements.txt!")
    requirements = get_requirements()
    log_txt(trainer.logger.experiment, trainer.logger.version, requirements, "requirements.txt")

    logging.log(logging.INFO,f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Upload Best Checkpoint to logging servers
    if config.upload_best_model and trainer.checkpoint_callback.best_model_path:
        logging.log(logging.INFO, f"Uploading best model to tracking server...")
        log_model_torch(trainer.logger.experiment, trainer.logger.version,
                        model_lit.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, map_location='cpu'))

    # Upload Artifacts to logging servers
    logging.log(logging.INFO, "Uploading artifacts to tracking server...")
    for artifact in os.listdir(trainer.log_dir):
        artifact_path = os.path.join(trainer.log_dir, artifact)
        # Log all except checkpoints
        if trainer.checkpoint_callback and artifact_path == trainer.checkpoint_callback.dirpath:
            if not config.upload_checkpoints:
                continue
        log_artifact(trainer.logger.experiment, trainer.logger.version, artifact_path)


    logging.log(logging.INFO, "Finalizing!")
    trainer.logger.finalize(status='FINISHED')


    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]