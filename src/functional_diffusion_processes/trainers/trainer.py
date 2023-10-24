import abc
import gc
import io
import logging
import os
from typing import Any, Callable, Tuple, Union

import flax
import flax.jax_utils as flax_utils
import hydra.utils

# import imageio
import jax
import numpy as np
import tensorflow as tf
import wandb
from cleanfid import fid
from flax import linen, traverse_util
from flax.training import checkpoints
from flax.training.checkpoints import restore_checkpoint
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from ..datasets import AudioDataset, ImageDataset
from ..datasets.base_dataset import BaseDataset
from ..losses.base_loss import Loss
from ..metrics import FIDMetric
from ..samplers import Sampler
from ..sdetools.base_sde import SDE
from ..utils.common import filter_mask, make_grid_image, process_images, save_samples, to_grayscale
from ..utils.scaler import get_data_inverse_scaler, get_data_scaler
from ..utils.training_state import TrainState
from .helpers import colorizing_fn, construct_sampling_fn, construct_train_step, inpainting_fn, sampling_fn

# import imageio


pylogger = logging.getLogger(__name__)


class Trainer(abc.ABC):
    """Class for training a model."""

    def __init__(
        self,
        mode: str,
        model_name: str,
        training_config: DictConfig,
        optimizer,
        evaluation_config: DictConfig,
        trainer_logging: DictConfig,
        sampler: Sampler,
        loss_obj: Loss,
    ) -> None:
        """Initialize a Trainer instance with configurations and core components.

        Args:
            mode (str): Specifies the mode of the trainer which can be either "train" or "eval".
            model_name (str): The name identifier for the model.
            training_config (DictConfig): A configuration dictionary for training settings.
            optimizer: The optimizer instance used for training.
            evaluation_config (DictConfig): A configuration dictionary for evaluation settings.
            trainer_logging (DictConfig): A configuration dictionary for logging settings.
            sampler (Sampler): A sampler instance for sampling from the model.
            loss_obj (Loss): A loss object used for computing the loss during training.
        """
        self.mode = mode
        self.model_name = model_name
        self.training_config = training_config
        self.optimizer = hydra.utils.instantiate(optimizer)
        self.evaluation_config = evaluation_config
        self.logging = trainer_logging
        self.sampler = sampler
        self.loss_obj = loss_obj
        self.checkpoint_dir = os.path.join(self.training_config.save_dir, self.training_config.checkpoint_dir)
        self.sample_dir = os.path.join(self.training_config.save_dir, self.training_config.sample_dir)
        self.eval_dir = os.path.join(self.training_config.save_dir, self.evaluation_config.eval_dir)

        # Create the directories for saving samples and checkpoints
        tf.io.gfile.makedirs(self.checkpoint_dir)
        tf.io.gfile.makedirs(self.sample_dir)
        tf.io.gfile.makedirs(self.eval_dir)
        tf.io.gfile.makedirs(os.path.join(self.eval_dir, "clean"))

    def initialize_wandb(
        self, dataset_config: DictConfig, sde_config: DictConfig, model_config: DictConfig
    ) -> Union[Run, RunDisabled, None]:
        """Initialize wandb if logging is enabled."""
        if self.logging.use_wandb:
            run = wandb.init(
                name=os.path.basename(self.logging.wandb_init.name),
                project=self.logging.wandb_init.project,
                entity=self.logging.wandb_init.entity,
                save_code=self.logging.wandb_init.save_code,
                config={
                    **self.training_config,
                    **dataset_config,
                    **sde_config,
                    **model_config,
                },
            )
        else:
            run = None
        return run

    def initialize_run(self, model, ds_train, sde):
        """Perform all initialization steps required for training."""
        run = self.initialize_wandb(ds_train.data_config, sde.sde_config, model.model_config)
        scaler = get_data_scaler(is_centered=ds_train.data_config.data_centered)
        inverse_scaler = get_data_inverse_scaler(is_centered=ds_train.data_config.data_centered)
        rng = jax.random.PRNGKey(seed=self.training_config.seed)
        rng, step_rng = jax.random.split(rng)
        batch_input = model.initialize_input(
            (ds_train.data_config.batch_size, *sde.sde_config.shape, ds_train.data_config.output_size)
        )
        params = jax.jit(model.initialize_model, backend="cpu")(step_rng, batch_input)
        flat_params = traverse_util.flatten_dict(params).values()
        tot_params = sum([jnp.size(p) for p in flat_params])
        pylogger.info("Total number of parameters: {:.2f}M".format(tot_params / 1e6))

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=self.optimizer,
            opt_state_params=self.optimizer.init(params),
            rng=rng,
            ema_params=params,
        )
        train_step_fn = construct_train_step(self.optimizer, self.loss_obj.construct_loss_fn(model))
        sample_fn = construct_sampling_fn(model, self.sampler)

        # Resume training when intermediate checkpoints are detected
        if self.training_config.resume_training:
            pylogger.warning("Resuming training from the latest checkpoint.")
            if self.logging.use_wandb and self.model_name != "local":
                model_file = wandb.use_artifact(self.model_name).download()
                state = restore_checkpoint(ckpt_dir=model_file, prefix="checkpoint_", target=state)
            else:
                state = checkpoints.restore_checkpoint(ckpt_dir=self.checkpoint_dir, target=state)

        return run, scaler, inverse_scaler, rng, state, train_step_fn, sample_fn, batch_input

    def train_step(
        self,
        train_step_fn: Callable,
        carry_state: Tuple,
        batch: jnp.ndarray,
        batch_input: jnp.ndarray,
    ) -> Tuple:
        """Perform a single training step, updating the model parameters.

        Args:
            train_step_fn (Callable): The train step function.
            carry_state (Tuple): The current state of the model and optimizer.
            batch (jnp.ndarray): The batch of data used for training.
            batch_input (jnp.ndarray): The input data to the model.

        Returns:
            Tuple: The updated state after performing the training step.
        """
        (rng, state) = carry_state

        (
            new_rng,
            loss,
            loss_inner,
            new_params,
            new_optim_state,
            batch_reconstructed,
            batch_corrupted,
            target,
        ) = train_step_fn(
            rng,
            state.params,
            state.opt_state_params,
            state.step,
            batch_input,
            batch,
        )
        ema_rate = self.training_config.ema_rate
        new_params_ema = jax.tree_map(
            lambda p_ema, p: p_ema * ema_rate + p * (1.0 - ema_rate),
            state.ema_params,
            new_params,
        )

        # update the state
        new_state = state.replace(
            rng=flax.jax_utils.unreplicate(new_rng),
            step=state.step + 1,
            opt_state_params=new_optim_state,
            params=new_params,
            ema_params=new_params_ema,
        )
        new_carry_state = (new_rng, new_state)

        loss = flax.jax_utils.unreplicate(loss)

        step = int(flax_utils.unreplicate(state.step))

        # Log the training progress
        if jax.host_id() == 0 and step % self.training_config.log_freq == 0:
            pylogger.info("step: %d, training_loss: %.5e" % (step, loss))
            if self.logging.use_wandb:
                wandb.log({"step": step, "loss": loss}, step=step)
            if loss_inner is not None:
                loss_inner = flax.jax_utils.unreplicate(loss_inner)
                for inner_step, loss in enumerate(loss_inner):
                    pylogger.info("step: %d, training_loss_inner: %.5e" % (step, loss))
                    if self.logging.use_wandb:
                        wandb.log({"step": step, f"loss inner step {inner_step}": loss}, step=step)

        return new_carry_state, batch_reconstructed, batch_corrupted, target

    def save_checkpoint(self, step, run, state):
        pylogger.info("Saving the model at step %d." % (step,))
        # Log the evaluation progress
        # Save the model parameters
        (
            params,
            opt_state_params,
            step_,
            ema_params,
        ) = flax_utils.unreplicate(
            (
                state.params,
                state.opt_state_params,
                state.step,
                state.ema_params,
            )
        )
        saved_state = state.replace(
            step=step_,
            opt_state_params=opt_state_params,
            params=params,
            ema_params=ema_params,
        )
        checkpoint_file = checkpoints.save_checkpoint(
            self.checkpoint_dir,
            saved_state,
            step=step_ // self.training_config.eval_freq,
            keep=np.inf,
        )
        if self.logging.use_wandb:
            wandb_model_artifact_name = str(step_) + "_" + run.id
            wandb_model = wandb.Artifact(wandb_model_artifact_name, type="model")
            wandb_model.add_file(checkpoint_file)
            run.log_artifact(wandb_model)

    # noinspection PyProtectedMember
    def train(self, model: linen.Module, ds_train: BaseDataset, sde: SDE) -> None:
        """Train the model with optional evaluation and logging.

        This method encapsulates the entire training process including initialization,
        training loop, checkpointing, evaluation, and logging. It supports different
        sampling types like colorization, inpainting, super resolution, and deblurring.

        Args:
            model (linen.Module): The model to be trained.
            ds_train (BaseDataset): The training dataset.
            sde (SDE): Stochastic differential equation object, governing the dynamics
                       for sampling.

        Raises:
            ValueError: If an unsupported dataset type is provided.

        Note:
            The method leverages the Weights & Biases (wandb) platform for logging
            and checkpointing, make sure it's configured properly if logging is enabled.
        """
        run, scaler, inverse_scaler, rng, state, train_step_fn, sample_fn, batch_input = self.initialize_run(
            model, ds_train, sde
        )
        # `state.step` is JAX integer on the GPU/TPU devices
        start_step = int(state.step)
        rng = state.rng

        # Replicate the train state on all devices
        (
            p_params,
            p_opt_state_params,
            p_step,
            p_ema_params,
            p_batch_input,
        ) = flax_utils.replicate(
            (
                state.params,
                state.opt_state_params,
                state.step,
                state.ema_params,
                batch_input,
            )
        )

        # update the TrainState with replicated parameters and optimizer state
        state = state.replace(
            params=p_params,
            opt_state_params=p_opt_state_params,
            step=p_step,
            ema_params=p_ema_params,
        )

        if jax.host_id() == 0:
            pylogger.info("Starting training loop at step %d." % (start_step,))

        rng = jax.random.fold_in(rng, jax.host_id())
        assert (
            self.training_config.log_freq % self.training_config.n_jitted_steps == 0
            and self.training_config.eval_freq % self.training_config.n_jitted_steps == 0
        ), "Missing logs or checkpoints!"
        ds_train_iter = iter(ds_train)

        with tqdm(
            total=self.training_config.total_steps + 1,
            initial=start_step,
            position=0,
            leave=True,
        ) as pbar:
            for step in range(
                start_step,
                self.training_config.total_steps + 1,
                self.training_config.n_jitted_steps,
            ):
                # Get the next batch of data and scale it
                batch = jax.tree_map(f=lambda x: scaler(x._numpy()), tree=next(ds_train_iter)["data"])
                if not self.training_config.sampling_only:
                    # Split the random number generator for the current step
                    rng, *next_rng = jax.random.split(key=rng, num=jax.local_device_count() + 1)
                    next_rng = jnp.asarray(next_rng)

                    ((_, state), batch_reconstructed, batch_corrupted, target) = self.train_step(
                        train_step_fn=train_step_fn,
                        carry_state=(next_rng, state),
                        batch=batch,
                        batch_input=p_batch_input,
                    )

                if not self.training_config.sampling_only and (
                    (jax.host_id() == 0 and step % self.training_config.checkpoint_freq == 0 and step != 0)
                ):
                    self.save_checkpoint(step, run, state)

                # Evaluate the model
                if self.training_config.sampling and (step % self.training_config.eval_freq == 0):
                    # if step != 0:
                    if jax.host_id() == 0:
                        pylogger.info("Generating samples at step %d." % (step,))

                    _, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)

                    _, b, g, c = batch.shape

                    sample_rng = jnp.asarray(sample_rng)

                    if self.training_config.sampling_type == "full":
                        batch_sampled, batch_sampled_last, batch_sampled_all = sampling_fn(
                            sample_fn, (sample_rng, state), p_batch_input
                        )
                    elif self.training_config.sampling_type == "colorization":
                        batch_grayscale = to_grayscale(batch)
                        batch_grayscale = batch_grayscale.reshape(-1, b, g, 1)
                        batch_sampled, batch_sampled_last, batch_sampled_all = colorizing_fn(
                            sample_fn, (sample_rng, state), p_batch_input, batch_grayscale
                        )
                    elif self.training_config.sampling_type == "inpainting":
                        config_object = OmegaConf.create(
                            {
                                "_target_": "functional_diffusion_processes.datasets.mnist_dataset.MNISTDataset",
                                "data_config": {
                                    "seed": 42,
                                    "batch_size": ds_train.data_config.batch_size,
                                    "image_height_size": ds_train.data_config.image_height_size,
                                    "image_width_size": ds_train.data_config.image_width_size,
                                    "output_size": 1,
                                    "random_flip": False,
                                    "uniform_dequantization": False,
                                    "data_centered": False,
                                    "data_dir": "${oc.env:DATA_ROOT}/tensorflow_datasets",
                                    "download": True,
                                    "is_mask": True,
                                },
                                "split": "train",
                                "evaluation": False,
                            }
                        )
                        ds_mask = hydra.utils.instantiate(config_object, _recursive_=False)
                        ds_mask_iter = iter(ds_mask)
                        batch_masked = jax.tree_map(f=lambda x: x._numpy(), tree=next(ds_mask_iter)["data"])
                        batch_sampled, batch_sampled_last, batch_sampled_all = inpainting_fn(
                            sample_fn, (sample_rng, state), p_batch_input, (batch * batch_masked), batch_masked
                        )

                    elif self.training_config.sampling_type == "deblurring":
                        n_rows, n_cols = ds_train.data_config.image_height_size, ds_train.data_config.image_width_size
                        batch_masked = filter_mask(batch.reshape(-1, b, n_rows, n_cols, c).shape, radius=10)
                        batch_freq = jnp.fft.fftshift(
                            jnp.fft.fft2(batch.reshape(-1, b, n_rows, n_cols, c), axes=(2, 3)),
                            axes=(2, 3),
                        )

                        batch_freq = batch_freq * batch_masked

                        batch_blurred = jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(batch_freq, axes=(2, 3)), axes=(2, 3)))

                        batch_blurred = batch_blurred.reshape(-1, b, g, c)

                        batch_masked = batch_masked.reshape(-1, b, g, c)

                        batch_sampled, batch_sampled_last, batch_sampled_all = inpainting_fn(
                            sample_fn, (sample_rng, state), p_batch_input, batch_blurred, batch_masked
                        )

                    if jax.host_id() == 0 and self.logging.use_wandb:
                        if isinstance(ds_train, ImageDataset):
                            this_sample_dir = os.path.join(
                                self.sample_dir,
                                "iter_{}_host_{}".format(step, jax.host_id()),
                            )
                            tf.io.gfile.makedirs(this_sample_dir)

                            # code below to show the gif of the sampled images
                            # processed_images = []
                            # for n in range(batch_sampled_all.shape[1]):
                            #     batch_sampled_i = batch_sampled_all[:, n, :, :, :]
                            #     batch_sampled_i = ds_train.postprocess_fn(
                            #         batch_data=batch_sampled_i, inverse_scaler=inverse_scaler
                            #     )
                            #     processed_images.append(np.asarray(batch_sampled_i))
                            #
                            # # Log the sampled images as a GIF
                            # imageio.mimwrite(
                            #     os.path.join(this_sample_dir, "image_sequence.gif"),
                            #     processed_images,
                            #     fps=10,
                            # )
                            # gif_wandb = wandb.Image(
                            #     os.path.join(this_sample_dir, "image_sequence.gif"),
                            #     caption="Sampled_all_gif",
                            # )
                            # wandb.log({"Sampled_all_gif": gif_wandb}, step=step)

                        batch_sampled = ds_train.postprocess_fn(batch_data=batch_sampled, inverse_scaler=inverse_scaler)
                        batch_sampled_last = ds_train.postprocess_fn(
                            batch_data=batch_sampled_last, inverse_scaler=inverse_scaler
                        )
                        batch_real = ds_train.postprocess_fn(
                            batch_data=batch.reshape(-1, b, g, c), inverse_scaler=inverse_scaler
                        )
                        if not self.training_config.sampling_only:
                            batch_target = ds_train.postprocess_fn(
                                batch_data=target.reshape(-1, b, g, c), inverse_scaler=inverse_scaler
                            )
                        if isinstance(ds_train, ImageDataset):
                            data_sampled = wandb.Image(np.asarray(batch_sampled), caption="Sampled")
                            data_sampled_rec = wandb.Image(np.asarray(batch_sampled_last), caption="Sampled Rec")
                            data_real = wandb.Image(np.asarray(batch_real), caption="Real")
                            if not self.training_config.sampling_only:
                                data_target = wandb.Image(np.asarray(batch_target), caption="Target")

                        elif isinstance(ds_train, AudioDataset):
                            sample_rate = ds_train.data_config.audio_sample_rate
                            long_audio_sampled = np.concatenate(
                                np.asarray(batch_sampled).reshape(-1, sample_rate), axis=0
                            )
                            data_sampled = wandb.Audio(long_audio_sampled, sample_rate=sample_rate, caption="Sampled")
                            if not self.training_config.sampling_only:
                                long_audio_target = np.concatenate(
                                    np.asarray(batch_target).reshape(-1, sample_rate), axis=0
                                )
                                data_target = wandb.Audio(long_audio_target, sample_rate=sample_rate, caption="Target")
                            long_audio_batch_sampled_rec = np.concatenate(
                                np.asarray(batch_sampled_last).reshape(-1, sample_rate), axis=0
                            )
                            data_sampled_rec = wandb.Audio(
                                long_audio_batch_sampled_rec, sample_rate=sample_rate, caption="Sampled Rec"
                            )
                            long_audio_batch_real = np.concatenate(
                                np.asarray(batch_real).reshape(-1, sample_rate), axis=0
                            )
                            data_real = wandb.Audio(long_audio_batch_real, sample_rate=sample_rate, caption="Real")

                        else:
                            raise ValueError("Unsupported dataset type: {}".format(type(ds_train)))

                        wandb.log({"Sampled": data_sampled}, step=step)
                        if not self.training_config.sampling_only:
                            wandb.log({"Target": data_target}, step=step)
                        wandb.log({"Sampled_rec": data_sampled_rec}, step=step)
                        wandb.log({"Real": data_real}, step=step)

                        if self.training_config.sampling_type == "colorization":
                            batch_gray = make_grid_image(
                                batch_grayscale.reshape(
                                    -1,
                                    ds_train.data_config.image_width_size,
                                    ds_train.data_config.image_height_size,
                                    1,
                                ),
                                inverse_scaler=inverse_scaler,
                            )
                            image_gray = wandb.Image(np.asarray(batch_gray), caption="Gray")
                            wandb.log({"Gray": image_gray}, step=step)
                        elif self.training_config.sampling_type == "inpainting":
                            batch_masked = make_grid_image(
                                ndarray=process_images(images=batch_masked * batch - (1 - batch_masked)),
                                inverse_scaler=inverse_scaler,
                            )
                            image_masked = wandb.Image(np.asarray(batch_masked), caption="Masked")
                            wandb.log({"Masked": image_masked}, step=step)
                        elif self.training_config.sampling_type == "deblurring":
                            batch_blurred = make_grid_image(
                                ndarray=process_images(images=batch_blurred),
                                inverse_scaler=inverse_scaler,
                            )
                            image_blurred = wandb.Image(np.asarray(batch_blurred), caption="Blurred")
                            wandb.log({"Blurred": image_blurred}, step=step)

                    if not self.training_config.sampling_only:
                        batch_reconstructed = ds_train.postprocess_fn(
                            batch_data=batch_reconstructed.reshape(-1, b, g, c), inverse_scaler=inverse_scaler
                        )
                        batch_corrupted = ds_train.postprocess_fn(
                            batch_data=batch_corrupted.reshape(-1, b, g, c), inverse_scaler=inverse_scaler
                        )
                        batch_real = ds_train.postprocess_fn(
                            batch_data=batch.reshape(-1, b, g, c), inverse_scaler=inverse_scaler
                        )
                        if isinstance(ds_train, ImageDataset):
                            data_reconstructed = wandb.Image(np.asarray(batch_reconstructed), caption="Reconstructed")
                            data_corrupted = wandb.Image(np.asarray(batch_corrupted), caption="Corrupted")
                            data_real = wandb.Image(np.asarray(batch_real), caption="Real")
                        elif isinstance(ds_train, AudioDataset):
                            sample_rate = ds_train.data_config.audio_sample_rate
                            long_audio_batch_reconstructed = np.concatenate(
                                np.asarray(batch_reconstructed).reshape(-1, sample_rate), axis=0
                            )
                            data_reconstructed = wandb.Audio(
                                long_audio_batch_reconstructed, sample_rate=sample_rate, caption="Reconstructed"
                            )
                            long_audio_batch_corrupted = np.concatenate(
                                np.asarray(batch_corrupted).reshape(-1, sample_rate), axis=0
                            )
                            data_corrupted = wandb.Audio(
                                long_audio_batch_corrupted, sample_rate=sample_rate, caption="Corrupted"
                            )
                            long_audio_batch_real = np.concatenate(
                                np.asarray(batch_real).reshape(-1, sample_rate), axis=0
                            )
                            data_real = wandb.Audio(long_audio_batch_real, sample_rate=sample_rate, caption="Real")
                        else:
                            raise ValueError("Unsupported dataset type: {}".format(type(ds_train)))

                        wandb.log({"Reconstructed": data_reconstructed}, step=step)
                        wandb.log({"Corrupted": data_corrupted}, step=step)
                        wandb.log({"Real": data_real}, step=step)
                # Update the progress bar
                pbar.update()

        wandb.finish()

    def evaluate(
        self,
        model: flax.linen.Module,
        ds_test: BaseDataset,
        fid_metric: FIDMetric,
        sde: SDE,
    ) -> None:
        """Evaluate the model on the test dataset.

        Args:
            model: The model to be evaluated.
            ds_test: The test dataset.
            fid_metric: The FID metric.
            sde: The SDE.
        """
        run, _, inverse_scaler, rng, state, _, sample_fn, batch_input = self.initialize_run(model, ds_test, sde)

        # Replicate the train state on all devices
        (p_ema_params, p_params, p_opt_state_params, p_step, p_batch_input) = flax_utils.replicate(
            (state.ema_params, state.params, state.opt_state_params, state.step, batch_input)
        )

        # update the TrainState with replicated parameters and optimizer state
        state = state.replace(
            params=p_params,
            opt_state_params=p_opt_state_params,
            step=p_step,
            ema_params=p_ema_params,
        )

        # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
        rng = jax.random.fold_in(rng, jax.host_id())

        # A data class for storing intermediate results to resume evaluation after pre-emption
        @flax.struct.dataclass
        class EvalMeta:
            sampling_round_id: int
            rng: Any

        num_sampling_rounds = (
            self.evaluation_config.num_samples // (ds_test.data_config.batch_size * jax.device_count()) + 1
        )

        # Restore evaluation after pre-emption
        eval_meta = EvalMeta(sampling_round_id=-1, rng=rng)
        eval_meta = checkpoints.restore_checkpoint(self.eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")
        if eval_meta.sampling_round_id < num_sampling_rounds - 1:
            begin_sampling_round = eval_meta.sampling_round_id + 1
        else:
            begin_sampling_round = 0
        rng = eval_meta.rng

        if jax.host_id() == 0:
            pylogger.info("Starting sampling loop at step %d." % (begin_sampling_round,))
        # Create a progress bar for tracking the training progress
        pbar = tqdm(
            total=num_sampling_rounds,
            initial=begin_sampling_round,
            position=0,
            leave=True,
        )

        for i in range(begin_sampling_round, num_sampling_rounds):
            if jax.host_id() == 0:
                pylogger.info("sampling -- round: %d" % i)
            this_sample_dir = os.path.join(self.eval_dir, f"ckpt_host_{jax.host_id()}")
            tf.io.gfile.makedirs(this_sample_dir)

            rng, *rng_s = jax.random.split(rng, jax.device_count() + 1)

            rng_s = jnp.asarray(rng_s)

            if not tf.io.gfile.exists(os.path.join(self.eval_dir, f"samples_{i}.npz")):
                batch_sampled, batch_sampled_last, _ = sampling_fn(sample_fn, (rng_s, state), p_batch_input)

                samples = inverse_scaler(batch_sampled_last)
                if isinstance(ds_test, ImageDataset):
                    samples = np.clip(samples * 255.0, 0, 255).astype(np.uint8)
                    samples = process_images(images=samples)
                elif isinstance(ds_test, AudioDataset):
                    min_intensity = ds_test.data_config.min_intensity
                    max_intensity = ds_test.data_config.max_intensity
                    samples = np.clip(
                        samples * (max_intensity - min_intensity) + min_intensity, min_intensity, max_intensity
                    )
                save_samples(
                    round_num=i,
                    samples=samples,
                    file_path=self.eval_dir,
                )
                if jax.host_id() == 0 and self.logging.use_wandb:
                    batch_sampled_last = ds_test.postprocess_fn(
                        batch_data=batch_sampled_last, inverse_scaler=inverse_scaler
                    )
                    if isinstance(ds_test, ImageDataset):
                        data_sampled = wandb.Image(np.asarray(batch_sampled_last), caption="Sampled")
                    elif isinstance(ds_test, AudioDataset):
                        sample_rate = ds_test.data_config.audio_sample_rate
                        long_audio_batch_sampled = np.concatenate(
                            np.asarray(batch_sampled_last).reshape(-1, sample_rate), axis=0
                        )
                        data_sampled = wandb.Audio(long_audio_batch_sampled, caption="Sampled", sample_rate=sample_rate)
                    else:
                        raise ValueError("Dataset type not supported for logging to wandb")
                    wandb.log({"Sampled": data_sampled}, step=i)
            else:
                pylogger.info("Skipping samples round %d as it already exists." % i)
                with tf.io.gfile.GFile(os.path.join(self.eval_dir, f"samples_{i}.npz"), "rb") as fin:
                    io_buffer = io.BytesIO(fin.read())
                    samples = np.load(io_buffer)["samples"]

            fid_metric.compute_and_store_generated_features(samples, self.eval_dir, i)

            gc.collect()

            eval_meta = eval_meta.replace(sampling_round_id=i, rng=rng)

            if i < num_sampling_rounds - 1:
                checkpoints.save_checkpoint(self.eval_dir, eval_meta, step=i, keep=1, prefix=f"meta_{jax.host_id()}_")
            # Update the progress bar
            pbar.update()
        if jax.host_id() == 0:
            # Compute FID
            fid_score, inception_score = fid_metric.compute_fid(self.eval_dir, num_sampling_rounds)
            pylogger.info("FID: %.6e" % fid_score)
            pylogger.info("Inception score %.6e" % inception_score)

            # Compute FID clean
            clean_dataset = os.path.join(
                fid_metric.metric_config.real_features_path, f"{fid_metric.metric_config.dataset_name.lower()}_clean"
            )
            clean_fake_dataset = os.path.join(self.eval_dir, "clean")
            fid_clean = fid.compute_fid(clean_fake_dataset, clean_dataset, mode="clean")
            pylogger.info("FID clean: %.6e" % fid_clean)

            # Compute FID-CLIP
            fid_clip = fid.compute_fid(clean_fake_dataset, clean_dataset, mode="clean", model_name="clip_vit_b_32")
            pylogger.info("FID-CLIP: %.6e" % fid_clip)

            if self.logging.use_wandb:
                wandb.log({"FID": float(fid_score)})
                wandb.log({"inception score": float(inception_score)})
                wandb.log({"FID clean": float(fid_clean)})
                wandb.log({"FID-CLIP": float(fid_clip)})
        wandb.finish()
