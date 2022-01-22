"""
 File name   : print_config.py
 Description : description

 Date created : 01.10.2021
 Author:  Ihar Khakholka
"""

from typing import List, Sequence

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from  copy import deepcopy

@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "callbacks",
        "datamodule",
        "eval",
        "mode",
        "model",
        "trainer",

        "work_dir",
        "test_after_training",
        "seed",
        "name",
        "upload_best_model"
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    config_copy = deepcopy(config)
    if 'augmentations' in config_copy.datamodule:
        config_copy.datamodule.augmentations = '...'

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config_copy.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)
