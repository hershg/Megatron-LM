import torch

from megatron.core.activations import apply_glu_linear_offset


def test_apply_glu_linear_offset_reuses_zero_offset_input() -> None:
    x = torch.tensor([1.0, 2.0])

    result = apply_glu_linear_offset(x, 0.0)

    assert result is x


def test_apply_glu_linear_offset_adds_nonzero_offset() -> None:
    x = torch.tensor([1.0, 2.0])

    result = apply_glu_linear_offset(x, 0.5)

    torch.testing.assert_close(result, torch.tensor([1.5, 2.5]))
    assert result is not x
