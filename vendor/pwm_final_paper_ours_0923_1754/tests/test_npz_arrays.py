import numpy as np
import pytest

from data.npz_arrays import npz_array


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("kind", ["float", "fortran", "scalar", "empty", "unicode", "bool"])
def test_member_reader_preserves_exact_arrays(tmp_path, compressed, kind):
    values = {"float": np.arange(60, dtype=np.float32).reshape(5, 4, 3),
              "fortran": np.asfortranarray(np.arange(20, dtype=np.float64).reshape(4, 5)),
              "scalar": np.asarray(.025), "empty": np.empty((0, 3), dtype=np.int64),
              "unicode": np.asarray(["right_hand", "thumb"]), "bool": np.asarray([True, False])}
    path = tmp_path / "episode.npz"
    (np.savez_compressed if compressed else np.savez)(path, **{"state/array": values[kind]})
    actual = npz_array(path, "state/array")
    assert actual.dtype == values[kind].dtype
    np.testing.assert_array_equal(actual, values[kind])
    if not compressed and actual.size:
        assert isinstance(actual, np.memmap) and not actual.flags.writeable


@pytest.mark.parametrize("compressed", [False, True])
def test_member_reader_never_unpickles_objects(tmp_path, compressed):
    path = tmp_path / "objects.npz"
    (np.savez_compressed if compressed else np.savez)(path, unsafe=np.array([dict(x=1)], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        npz_array(path, "unsafe")


def test_member_reader_missing_key_fails(tmp_path):
    path = tmp_path / "episode.npz"
    np.savez(path, x=np.arange(5))
    with pytest.raises(KeyError):
        npz_array(path, "not_a_field")
