# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.decode_benchmark import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkDecodeSubcommand(BenchmarkSubcommandBase):
    """The `decode` subcommand for `vllm bench`.

    Benchmarks decode performance by replaying captured CUDA/HIP graphs
    multiple times to measure kernel execution performance.
    """

    name = "decode"
    help = "Benchmark decode performance using captured CUDA/HIP graphs."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
