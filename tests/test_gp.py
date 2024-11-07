import torch
from useful.gp import GPDataset
import gunpowder as gp
from funlib.persistence import Array
from torch.utils.data import IterableDataset
import numpy as np
import pytest


@pytest.fixture
def pipeline():
    # Create a dummy pipeline
    source = gp.ArraySource(
        key=gp.ArrayKey("RAW"),
        array=Array(data=np.zeros((100, 100, 100), dtype=np.uint8)),
    )
    source += gp.PreCache(16)
    return source


@pytest.fixture
def gp_request():
    request = gp.BatchRequest(
        {
            gp.ArrayKey("RAW"): gp.ArraySpec(roi=gp.Roi((0, 0, 0), (10, 10, 10))),
        }
    )
    return request


def test_gp_dataset(pipeline, gp_request):
    dataset = GPDataset(pipeline=pipeline, request=gp_request)
    # Check that the dataset is iterable
    assert isinstance(dataset, IterableDataset)
    # Check that the dataset yields a sample
    sample = next(iter(dataset))
    assert isinstance(sample, dict)
    assert "RAW" in sample
    assert isinstance(sample["RAW"], np.ndarray)
    assert sample["RAW"].shape == (10, 10, 10)
    assert sample["RAW"].dtype == np.uint8
    assert np.all(sample["RAW"] == 0)


def test_gp_multiprocess(pipeline, gp_request):
    """Test that the dataset can be used with DataLoader."""
    from torch.utils.data import DataLoader

    dataset = GPDataset(pipeline=pipeline, request=gp_request)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=1)
    for sample in dataloader:
        assert isinstance(sample, dict)
        assert "RAW" in sample
        assert isinstance(sample["RAW"], torch.Tensor)
        assert sample["RAW"].shape == (16, 10, 10, 10)
        assert torch.all(sample["RAW"] == 0)
        break
