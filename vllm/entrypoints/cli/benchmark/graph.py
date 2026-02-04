# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Set environment variables BEFORE any vLLM imports to ensure logging
# goes to stderr so stdout is clean for JSON output.
import os

os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"

import argparse

from vllm.benchmarks.graph_benchmark import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkGraphSubcommand(BenchmarkSubcommandBase):
    """The `graph` subcommand for `vllm bench`.

    Benchmarks captured CUDA/HIP graphs by replaying them multiple times
    to measure kernel execution performance.
    """

    name = "graph"
    help = "Benchmark captured CUDA/HIP graphs for kernel performance analysis."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
