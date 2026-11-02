"""Tests that nmf_pytorch computes in float32.

The inputs are integer image data, so float32 carries every value exactly.
Most GPUs run float64 many times slower, and float64 also doubles the memory a
block needs. The caller must not be able to change that by passing float64.

The risk this guards against is silent: a float64 input would still fit and
still give an answer, only slower, and nothing in the output would say so.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentNMF.NMF_methods import nmf_pytorch  # noqa: E402


def toy(dtype=np.float32, n_vox=12, n_time=8, k=2, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.random((n_vox, k))
    H = rng.random((k, n_time))
    return dict(
        V=(S @ H).astype(dtype),
        S_init=S.astype(dtype),
        H_init=H.astype(dtype),
        B=np.ones((n_vox, k), dtype=dtype),
        num_iterations=3,
        update_int=10 ** 6,
        min_iterations=0,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_result_is_float32_whatever_the_input(dtype):
    """float64 in must not give float64 out."""
    S, H, _ = nmf_pytorch(**toy(dtype))
    assert S.dtype == np.float32, "S came back as {}".format(S.dtype)
    assert H.dtype == np.float32, "H came back as {}".format(H.dtype)


def test_a_float64_input_gives_the_same_answer_as_float32():
    """The cast must not change the result, only the cost."""
    a, _, _ = nmf_pytorch(**toy(np.float32))
    b, _, _ = nmf_pytorch(**toy(np.float64))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
