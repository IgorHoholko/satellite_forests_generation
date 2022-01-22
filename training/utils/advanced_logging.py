"""
 File name   : high_level_logging.py
 Description : Logging unified advanced functional for different loggers

 Date created : 15.10.2021
 Author:  Ihar Khakholka
"""



import numpy as np
from typing import Union
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def log_txt(loggers: list, run_id: str, text: str, artifact_file: str) -> None:
    from mlflow.tracking.client import MlflowClient
    for logger in loggers:
        if logger.__class__ == MlflowClient:
            logger: MlflowClient = logger
            logger.log_text(run_id, text, artifact_file)
        else:
            raise RuntimeError(f"Not implemented <log_txt> method for {logger.__class__} logger!")

@rank_zero_only
def log_model_torch(loggers: list, run_id: str, model: LightningModule,
                    model_name: str = 'model') -> None:
    from mlflow.tracking.client import MlflowClient
    import mlflow
    for logger in loggers:
        if logger.__class__ == MlflowClient:
            mlflow.set_tracking_uri(logger._registry_uri)
            with mlflow.start_run(run_id) as run:
                mlflow.pytorch.log_model(model, model_name)
        else:
            raise RuntimeError(f"Not implemented <log_model_torch> method for {logger.__class__} logger!")

@rank_zero_only
def log_artifact(loggers: list, run_id: str, local_path: str, artifact_path: str = None) -> None:
    from mlflow.tracking.client import MlflowClient
    for logger in loggers:
        if logger.__class__ == MlflowClient:
            logger: MlflowClient = logger
            logger.log_artifact(run_id, local_path, artifact_path)
        else:
            raise RuntimeError(f"Not implemented <log_artifact> method for {logger.__class__} logger!")

@rank_zero_only
def log_figure(loggers: list, run_id: str, figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
               artifact_file: str) -> None:
    from mlflow.tracking.client import MlflowClient
    for logger in loggers:
        if logger.__class__ == MlflowClient:
            logger: MlflowClient = logger
            logger.log_figure(run_id, figure, artifact_file)
        else:
            raise RuntimeError(f"Not implemented <log_artifact> method for {logger.__class__} logger!")

@rank_zero_only
def log_image(loggers: list, run_id: str, image: np.ndarray,
               artifact_file: str) -> None:
    from mlflow.tracking.client import MlflowClient
    for logger in loggers:
        if logger.__class__ == MlflowClient:
            logger: MlflowClient = logger
            logger.log_image(run_id, image, artifact_file)
        else:
            raise RuntimeError(f"Not implemented <log_artifact> method for {logger.__class__} logger!")


