import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")


def pytest_collection_modifyitems(config, items):
    """Fix M2: skip @pytest.mark.gpu tests on CPU-only machines.

    When ``torch.cuda.is_available()`` returns False, apply a skip marker to
    every item that carries the ``gpu`` mark, so a plain ``pytest`` run on a
    CPU box skips them instead of erroring out.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        return  # GPU present — let them run normally

    skip_no_gpu = pytest.mark.skip(reason="GPU not available (torch.cuda.is_available() is False)")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip_no_gpu)
