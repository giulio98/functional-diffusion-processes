# MIT License
#
# Copyright (c) 2022 Fan Bao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from abc import ABC

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from . import BaseViT
from .blocks import Block
from .embeddings import AddPositionEmbs, get_timestep_embedding
from .embeddings.patch_embedding import PatchEmbeddings
from .encodings.position_encoding import AddPositionEncodings


# noinspection PyAttributeOutsideInit
class UViT(BaseViT, ABC):
    """Implementation of the UViT model, a variant of Vision Transformer (ViT) introduced.

    in the paper "All are Worth Words: A ViT Backbone for Diffusion Models" (https://arxiv.org/abs/2209.12152).

    The model employs patch embeddings, position embeddings/encodings, and transformer blocks to process
    the input and return a processed tensor.

    Attributes:
        model_config (DictConfig): Configuration dictionary for setting up the model.

    Methods:
        setup(): Set up the VisionTransformer module with the provided configuration.
        reshape_input(x): Reshape the input tensor based on the model configuration.
        separate_data_from_time(x): Separate data and time information from the input tensor.
        unpatchify(x: jnp.ndarray): Convert patch embeddings back to image-like or sequence-like tensor.
        __call__(inputs: jnp.ndarray, *, train: bool): Process the input tensor through the UViT model.
    """

    model_config: DictConfig

    def setup(self):
        """Set up the VisionTransformer module based on the provided configuration in `model_config`."""
        self.patch_size = self.model_config["patch_size"]
        self.in_chans = self.model_config["in_chans"]
        self.transformer = self.model_config["transformer"]
        self.embeddings_size = self.model_config["embeddings_size"]
        self.image_size = self.model_config["image_size"]
        self.old_image_size = self.model_config["old_image_size"]
        self.is_unidimensional = self.model_config["is_unidimensional"]
        self.dtype = jnp.float32
        self.Block = Block
        if self.model_config["add_position"] == "embedding":
            self.add_position = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),
                name="posembed_input",
                old_image_size=self.old_image_size,
                patch_size=self.patch_size,
                image_size=self.image_size,
            )
        elif self.model_config["add_position"] == "encoding":
            self.add_position = AddPositionEncodings(
                num_hiddens=self.embeddings_size,
                patch_size=self.patch_size,
                old_image_size=self.old_image_size,
                image_size=self.image_size,
            )

    def reshape_input(self, x):
        """Reshape the input tensor based on whether the model is configured to be unidimensional or not.

        Args:
            x: Input tensor to be reshaped.

        Returns:
            Reshaped input tensor.
        """
        b, g, o = x.shape
        if self.is_unidimensional:
            return x.reshape(b, self.image_size, o)
        else:
            return x.reshape(b, self.image_size, self.image_size, o)

    def separate_data_from_time(self, x):
        """Separate data and time information from the input tensor.

        Args:
            x: Input tensor containing data and time information.

        Returns:
            Tuple containing separated data and time tensors.
        """
        if self.is_unidimensional:
            data = x[..., 1:-1]
            time = x[..., -1][:, 0]
        else:
            data = x[..., 2:-1]
            time = x[..., -1][:, 0, 0]
        return data, time

    def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert patch embeddings back to image-like or sequence-like tensor based on the model's configuration.

        Args:
            x (jnp.ndarray): Input tensor with patch embeddings.

        Returns:
            jnp.ndarray: Unpatchified tensor representing images or sequences.
        """
        if self.is_unidimensional:
            channels = 1
            patch_size = int((x.shape[2] // channels))
            h = int(x.shape[1])
            x = einops.rearrange(x, "B (h) (p1 C) -> B (h p1) C", h=h, p1=patch_size)
        else:
            channels = 3
            patch_size = int((x.shape[2] // channels) ** 0.5)
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
            x = einops.rearrange(x, "B (h w) (p1 p2 C) -> B (h p1) (w p2) C", h=h, p1=patch_size, p2=patch_size)
        return x

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """Process the input tensor through the UViT model.

        The method processes input data in patches, applies transformer blocks,
        and returns the processed tensor.

        Args:
            inputs (jnp.ndarray): Input tensor of shape (batch_size, grid_size, num_channels).
            train (bool): A flag indicating if the model is in training mode.

        Returns:
            jnp.ndarray: Processed tensor of shape (batch_size, grid_size, num_channels).
        """
        x = inputs
        b, g, o = x.shape
        x = self.reshape_input(x)
        # coordinates = x[..., :2]
        x, time = self.separate_data_from_time(x)
        # Add patch embeddings
        x = PatchEmbeddings(
            patch_size=self.patch_size,
            dtype=self.dtype,
            in_chans=self.in_chans,
            embed_dim=self.embeddings_size,
            is_unidimensional=self.is_unidimensional,
        )(x)
        # Add time embeddings
        time = time * 999
        emb_time = get_timestep_embedding(time, embedding_dim=self.embeddings_size)

        emb_time = nn.Dense(
            self.embeddings_size * 4,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(1, "fan_avg", "uniform"),
        )(emb_time)
        emb_time = nn.Dense(
            self.embeddings_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(1, "fan_avg", "uniform"),
        )(nn.silu(emb_time))
        time_token = jnp.expand_dims(emb_time, axis=1)
        x = jnp.concatenate([time_token, x], axis=1)
        # Add position embeddings or encodings
        x = self.add_position(x)
        x = nn.Dropout(rate=0.0)(x, deterministic=not train)

        skips = []
        for lyr in range(self.transformer.num_layers // 2):
            x = Block(
                mlp_dim=self.transformer.mlp_dim,
                mlp_ratio=self.transformer.mlp_ratio,
                dtype=self.dtype,
                dropout_rate=self.transformer.dropout_rate,
                attention_dropout_rate=self.transformer.attention_dropout_rate,
                name=f"in_block_{lyr}",
                num_heads=self.transformer.num_heads,
            )(x, deterministic=not train)
            skips.append(x)

        x = Block(
            mlp_dim=self.transformer.mlp_dim,
            mlp_ratio=self.transformer.mlp_ratio,
            dtype=self.dtype,
            dropout_rate=self.transformer.dropout_rate,
            attention_dropout_rate=self.transformer.attention_dropout_rate,
            name="mid_block_0",
            num_heads=self.transformer.num_heads,
        )(x, deterministic=not train)

        for lyr in range(self.transformer.num_layers // 2):
            x = Block(
                mlp_dim=self.transformer.mlp_dim,
                mlp_ratio=self.transformer.mlp_ratio,
                dtype=self.dtype,
                dropout_rate=self.transformer.dropout_rate,
                attention_dropout_rate=self.transformer.attention_dropout_rate,
                name=f"out_block_{lyr}",
                num_heads=self.transformer.num_heads,
                skip=self.transformer.skip,
            )(x, skip=skips.pop(), deterministic=not train)

        x = nn.LayerNorm(name="norm", dtype=self.dtype)(x)

        x = nn.Dense(
            features=self.patch_size**2 * self.in_chans
            if not self.is_unidimensional
            else self.patch_size * self.in_chans,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.constant(0),
            name="decoder_pred",
        )(x)
        x = x[:, 1:, :]  # remove time token
        x = self.unpatchify(x)
        x = nn.Conv(features=self.in_chans, dtype=self.dtype, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(x)
        return x.reshape(b, g, self.in_chans)
