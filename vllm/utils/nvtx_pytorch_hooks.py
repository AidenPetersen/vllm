# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Backward compatibility module for nvtx PyTorch hooks.

This module re-exports symbols from gpu_tracing_hooks for backward compatibility.
New code should import from vllm.utils.gpu_tracing_hooks instead.
"""

# Re-export all symbols from gpu_tracing_hooks for backward compatibility
from vllm.utils.gpu_tracing_hooks import (
    PytHooks,
    ResultHolder,
    construct_marker_dict_and_push,
    layerwise_nvtx_marker_context,
    layerwise_tracing_marker_context,
    print_tensor,
    process_layer_params,
)

__all__ = [
    "PytHooks",
    "ResultHolder",
    "construct_marker_dict_and_push",
    "layerwise_nvtx_marker_context",
    "layerwise_tracing_marker_context",
    "print_tensor",
    "process_layer_params",
]
