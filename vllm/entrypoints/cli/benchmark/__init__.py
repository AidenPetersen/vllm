# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Import all benchmark subcommands to register them with BenchmarkSubcommandBase
from vllm.entrypoints.cli.benchmark.graph import BenchmarkGraphSubcommand
from vllm.entrypoints.cli.benchmark.latency import BenchmarkLatencySubcommand
from vllm.entrypoints.cli.benchmark.mm_processor import (
    BenchmarkMMProcessorSubcommand,
)
from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
from vllm.entrypoints.cli.benchmark.startup import BenchmarkStartupSubcommand
from vllm.entrypoints.cli.benchmark.sweep import BenchmarkSweepSubcommand
from vllm.entrypoints.cli.benchmark.throughput import BenchmarkThroughputSubcommand

__all__ = [
    "BenchmarkGraphSubcommand",
    "BenchmarkLatencySubcommand",
    "BenchmarkMMProcessorSubcommand",
    "BenchmarkServingSubcommand",
    "BenchmarkStartupSubcommand",
    "BenchmarkSweepSubcommand",
    "BenchmarkThroughputSubcommand",
]
