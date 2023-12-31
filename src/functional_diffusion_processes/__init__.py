try:
    from ._version import __version__ as __version__

    __all__ = [
        "datasets",
        "metrics",
        "models",
        "samplers",
        "sdetools",
        "trainers",
        "utils",
    ]
except ImportError:
    import sys

    print(
        "Project not installed in the current env, activate the correct env or install it with:\n\tpip install -e .",
        file=sys.stderr,
    )
    __version__ = "unknown"
