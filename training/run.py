"""
 File name   : run_experiment.py
 Description : description

 Date created : 30.09.2021
 Author:  Ihar Khakholka
"""

import dotenv
import sys, os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

dotenv.load_dotenv(str(PROJECT_DIR/'.env'), override=True)

pythonpaths = [str(PROJECT_DIR / p) for p in os.environ['PYTHONPATH'].split(':')]
sys.path.extend(pythonpaths)

import hydra
from omegaconf import DictConfig

import cv2
cv2.setNumThreads(0) # Fix dataloader deadlock


@hydra.main(config_path= str(PROJECT_DIR/"configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from training.lit import train_lit
    from training.utils import print_config

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # Train model
    return train_lit(cfg)


if __name__ == "__main__":
    main()
