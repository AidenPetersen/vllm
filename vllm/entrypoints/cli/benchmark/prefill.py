# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.prefill_benchmark import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkPrefillSubcommand(BenchmarkSubcommandBase):
    """The `prefill` subcommand for `vllm bench`.

    Benchmarks prefill (prompt processing) performance by running the model
    forward pass in eager mode to measure execution latency for various
    prompt lengths.
    """

    name = "prefill"
    help = "Benchmark prefill performance (eager mode forward pass)."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
