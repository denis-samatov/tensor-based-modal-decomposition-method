"""Regression contracts for zero-reference errors and tensor-only file loading."""
import pytest
import torch

from TBMD.core.data.loaders import BaseDataLoader
from TBMD.core.decomposition.hosvd import TuckerDecomposer


@pytest.mark.parametrize("collection", [False, True])
def test_exact_zero_reconstruction_has_zero_error(collection):
    zero = torch.zeros(2, 2, 2)
    model = TuckerDecomposer({"zero": zero} if collection else zero, ranks=[1, 1, 1])
    model.decompose()
    model.reconstruct()
    error = model.reconstruction_errors["zero"] if collection else model.reconstruction_errors
    assert error == 0.0


@pytest.mark.parametrize("payload", [[], {"x": torch.ones(2)}])
def test_non_tensor_payload_has_explicit_error(tmp_path, payload):
    path = tmp_path / "data.pt"
    torch.save(payload, path)
    with pytest.raises((TypeError, ValueError), match="tensor|key"):
        BaseDataLoader(path).load_tensor()


def test_named_tensor_can_be_loaded_without_arbitrary_objects(tmp_path):
    path = tmp_path / "data.pt"
    torch.save({"x": torch.tensor([2, 3])}, path)
    assert BaseDataLoader(path).load_tensor("x").tolist() == [2.0, 3.0]


def test_nonzero_residual_with_zero_reference_is_undefined():
    from TBMD.core.decomposition.hosvd import ValidationError, _relative_reconstruction_error

    with pytest.raises(ValidationError, match="zero reference norm"):
        _relative_reconstruction_error(1.0, 0.0)


class UnexpectedPayload:
    """A harmless custom class is outside the tensor-only load contract."""


def test_tensor_loader_rejects_custom_pickle_objects(tmp_path):
    import pickle

    path = tmp_path / "unexpected.pt"
    torch.save(UnexpectedPayload(), path)
    with pytest.raises(pickle.UnpicklingError):
        BaseDataLoader(path).load_tensor()
