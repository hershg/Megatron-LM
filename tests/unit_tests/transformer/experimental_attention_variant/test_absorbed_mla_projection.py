# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    absorbed_mla as absorbed_mla_module,
)


@pytest.mark.parametrize("leading_shape", [(5,), (2, 3)])
def test_absorbed_k_up_projection_writes_final_layout_with_correct_gradients(
    leading_shape, monkeypatch
):
    """Absorbed K-up projection should match the reference without an intermediate layout."""
    torch.manual_seed(123)
    num_heads, qk_head_dim, kv_lora_rank, qk_pos_emb_head_dim = 3, 4, 5, 2
    query = torch.randn(
        *leading_shape,
        num_heads,
        qk_head_dim + qk_pos_emb_head_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_query = query.detach().clone().requires_grad_()
    k_up_weight = torch.randn(
        num_heads, qk_head_dim, kv_lora_rank, dtype=torch.float64, requires_grad=True
    )
    reference_weight = k_up_weight.detach().clone().requires_grad_()

    q_no_pe, q_pos_emb = torch.split(query, [qk_head_dim, qk_pos_emb_head_dim], dim=-1)
    reference_q_no_pe, reference_q_pos_emb = torch.split(
        reference_query, [qk_head_dim, qk_pos_emb_head_dim], dim=-1
    )
    assert not q_no_pe.is_contiguous()

    projection_output_storages = []
    torch_mm = torch.mm

    def record_projection_output_storage(*args, **kwargs):
        output = kwargs["out"]
        projection_output_storages.append(output.untyped_storage().data_ptr())
        return torch_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", record_projection_output_storage)
    actual = absorbed_mla_module._apply_absorbed_k_up_projection(
        q_no_pe, q_pos_emb, k_up_weight
    )
    expected = torch.cat(
        [
            torch.einsum(
                "...nd,ndk->...nk", reference_q_no_pe, reference_weight
            ).contiguous(),
            reference_q_pos_emb,
        ],
        dim=-1,
    )

    assert actual.is_contiguous()
    assert actual.shape == expected.shape
    assert projection_output_storages == [
        actual.untyped_storage().data_ptr()
    ] * num_heads
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    output_gradient = torch.randn_like(actual)
    actual.backward(output_gradient)
    expected.backward(output_gradient)
    torch.testing.assert_close(query.grad, reference_query.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        k_up_weight.grad, reference_weight.grad, rtol=1e-12, atol=1e-12
    )


def test_absorbed_k_up_projection_passes_gradcheck():
    """The direct-final absorbed projection should supply a correct first-order backward."""
    torch.manual_seed(123)
    q_no_pe = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    q_pos_emb = torch.randn(2, 2, 2, dtype=torch.float64, requires_grad=True)
    k_up_weight = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        absorbed_mla_module._apply_absorbed_k_up_projection,
        (q_no_pe, q_pos_emb, k_up_weight),
        rtol=1e-4,
        atol=1e-6,
    )
