# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, call

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


def _make_bucket():
    return SimpleNamespace(params_list=[object()])


def _make_optimizer(model_chunk, buffer):
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.model_chunks = [model_chunk]
    optimizer.per_model_buffers = {0: [buffer]}
    return optimizer


def test_chained_distopts_sync_only_bucket_groups_from_owned_buffers():
    dense_bucket = _make_bucket()
    expert_bucket = _make_bucket()
    dense_group = SimpleNamespace(buckets=[dense_bucket])
    expert_group = SimpleNamespace(buckets=[expert_bucket])
    start_param_sync = Mock()
    model_chunk = SimpleNamespace(
        bucket_groups=[dense_group],
        expert_parallel_bucket_groups=[expert_group],
        _start_bucket_group_param_sync=start_param_sync,
    )
    dense_optimizer = _make_optimizer(model_chunk, SimpleNamespace(buckets=[dense_bucket]))
    expert_optimizer = _make_optimizer(model_chunk, SimpleNamespace(buckets=[expert_bucket]))

    dense_optimizer.start_param_sync_for_bucket_group_subset()
    expert_optimizer.start_param_sync_for_bucket_group_subset()

    assert start_param_sync.call_args_list == [
        call(dense_group, force_sync=False),
        call(expert_group, force_sync=False),
    ]
