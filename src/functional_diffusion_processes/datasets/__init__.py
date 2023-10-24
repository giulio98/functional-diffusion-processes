from .base_dataset import BaseDataset  # isort:skip
from .audio_dataset import AudioDataset
from .celeba_dataset import CELEBADataset
from .image_dataset import ImageDataset
from .mnist_dataset import MNISTDataset
from .spoken_digit_dataset import SpokenDigitDataset

__all__ = [
    "BaseDataset",
    "ImageDataset",
    "AudioDataset",
    "CELEBADataset",
    "MNISTDataset",
    "SpokenDigitDataset",
]
