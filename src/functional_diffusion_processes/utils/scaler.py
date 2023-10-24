from typing import Callable


def get_data_scaler(is_centered: bool) -> Callable:
    """Normalize data. Assume data are always in [0, 1].

    Args:
        is_centered: boolean if True data will be centered in [-1, 1].
    """
    if is_centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(is_centered: bool) -> Callable:
    """Inverse data normalizer.

    Rescale data to original range at the end of the diffusion.

    Args:
        is_centered: boolean if True data will rescaled from [-1, 1] to [0, 1].
    """
    if is_centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x
