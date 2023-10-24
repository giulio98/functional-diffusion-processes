import logging

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

from nn_core.common import PROJECT_ROOT

# Force the execution of __init__.py if this file is executed directly.
import functional_diffusion_processes  # noqa

pylogger = logging.getLogger(__name__)

import os

import tensorflow as tf


def run(cfg: DictConfig) -> None:
    """Run Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    # Instantiate train and test datasets
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    pylogger.info(f"Instantiating <{cfg.datasets.train['_target_']}>")
    ds_train = hydra.utils.instantiate(cfg.datasets.train, _recursive_=False)

    pylogger.info(f"Instantiating <{cfg.datasets.test['_target_']}>")
    ds_test = hydra.utils.instantiate(cfg.datasets.test, _recursive_=False)

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.models['_target_']}>")
    model = hydra.utils.instantiate(cfg.models, _recursive_=False)

    # Instantiate sde
    pylogger.info(f"Instantiating <{cfg.sdes['_target_']}>")
    sde = hydra.utils.instantiate(cfg.sdes, _recursive_=False)

    # Instantiate corrector
    pylogger.info(f"Instantiating <{cfg.correctors['_target_']}>")
    corrector = hydra.utils.instantiate(
        OmegaConf.create(
            {k: v for k, v in OmegaConf.to_container(cfg.correctors, resolve=True).items() if k != "name"}
        ),
        sde=sde,
        _recursive_=False,
    )

    # Instantiate predictors
    pylogger.info(f"Instantiating <{cfg.predictors['_target_']}>")
    predictor = hydra.utils.instantiate(
        OmegaConf.create(
            {k: v for k, v in OmegaConf.to_container(cfg.predictors, resolve=True).items() if k != "name"}
        ),
        sde=sde,
        _recursive_=False,
    )

    # Instantiate sampler
    pylogger.info(f"Instantiating <{cfg.samplers['_target_']}>")
    sampler = hydra.utils.instantiate(
        cfg.samplers, predictor=predictor, corrector=corrector, sde=sde, _recursive_=False
    )

    # Instantiate the loss
    pylogger.info(f"Instantiating <{cfg.losses['_target_']}>")
    loss_obj = hydra.utils.instantiate(cfg.losses, sde=sde, _recursive_=False)

    # Instantiate trainer
    pylogger.info(f"Instantiating <{cfg.trainers['_target_']}>")

    trainer = hydra.utils.instantiate(cfg.trainers, loss_obj=loss_obj, sampler=sampler, _recursive_=False)

    if not cfg.trainers.mode == "eval":
        pylogger.info("Starting training!")
        trainer.train(model=model, ds_train=ds_train, sde=sde)
        pylogger.info("Training finished!")

    if cfg.trainers.mode in ["train_eval", "eval"]:
        # Instantiate the metric
        pylogger.info(f"Instantiating <{cfg.metrics['_target_']}>")
        metric = hydra.utils.instantiate(cfg.metrics, dataset=ds_test)

        pylogger.info("Starting testing!")
        trainer.evaluate(model=model, ds_test=ds_test, fid_metric=metric, sde=sde)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    """Run the main function."""
    run(cfg)


if __name__ == "__main__":
    logging.basicConfig(force=True, level=logging.INFO)
    main()
