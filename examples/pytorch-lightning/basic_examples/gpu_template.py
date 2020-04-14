"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pl_examples.models.lightning_template import LightningTemplateModel


from pathlib import Path
from composed_trackers import Config, build_from_cfg, TRACKERS
from composed_trackers.adapters.pytorch_lightning import adapt_to_pytorch_lightning


SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(hparams)

    # init Config
    #tracker = TestTubeLogger(save_dir='./logs')
    fn_config = Path('configs/example_gpu_00.yaml')
    cfg = Config.fromfile(fn_config)

    # init tracker
    params = vars(hparams)
    params.update(cfg.to_flatten_dict())  # flattening config --> params
    print(params)
    tracker = build_from_cfg(cfg.tracker, TRACKERS, 
                         {'params': params} 
                        )
    adapt_to_pytorch_lightning(tracker)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        logger=tracker,
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
        row_log_interval=cfg.tracker.row_log_interval,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
