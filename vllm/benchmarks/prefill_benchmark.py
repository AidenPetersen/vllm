# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark prefill performance (eager mode forward pass).

This module provides utilities for benchmarking the performance of prefill
(prompt processing) in vLLM. It runs the model forward pass in eager mode
(without CUDA graphs) to measure prefill latency for various prompt lengths.
"""

import argparse
import dataclasses
import json
import time
from typing import Any

import numpy as np

from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class PrefillBenchmarkResult:
    """Results from benchmarking prefill at a specific token count."""

    num_tokens: int
    """Number of tokens benchmarked."""

    num_iterations: int
    """Number of timed iterations run."""

    warmup_iterations: int
    """Number of warmup iterations run before timing."""

    mean_ms: float
    """Mean execution time in milliseconds."""

    median_ms: float
    """Median execution time in milliseconds."""

    std_ms: float
    """Standard deviation of execution time in milliseconds."""

    min_ms: float
    """Minimum execution time in milliseconds."""

    max_ms: float
    """Maximum execution time in milliseconds."""

    percentiles: dict[str, float]
    """Percentile values (p50, p90, p95, p99)."""

    all_times_ms: list[float]
    """List of all measured execution times."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_tokens": self.num_tokens,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "percentiles": self.percentiles,
            "all_times_ms": self.all_times_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrefillBenchmarkResult":
        """Create from dictionary."""
        return cls(**data)


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[int, dict[str, Any]]
) -> None:
    """Save results in PyTorch benchmark format."""
    metrics = {}
    for num_tokens, result in results.items():
        metrics[f"prefill_{num_tokens}_mean_ms"] = [result["mean_ms"]]
        metrics[f"prefill_{num_tokens}_median_ms"] = [result["median_ms"]]

    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics=metrics,
        extra_info={"results": {str(k): v for k, v in results.items()}},
    )
    if pt_records:
        import os

        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for prefill benchmarking."""
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of timed iterations per token size.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--prefill-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Token counts to benchmark for prefill. "
        "Example: --prefill-sizes 128 256 512 1024.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to send for initial warmup inference.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=32,
        help="Input length for initial warmup requests.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=16,
        help="Output length for initial warmup requests.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON.",
    )

    # Add engine args
    parser = EngineArgs.add_cli_args(parser)
    # Enable prefix caching disabled by default for benchmarking
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace) -> None:
    """Main function for prefill benchmarking CLI."""
    import os

    # Disable V1 multiprocessing to allow direct access to model components.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    logger.info(
        "Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 for direct model access."
    )

    # Lazy imports
    from vllm import LLM, SamplingParams

    engine_args = EngineArgs.from_cli_args(args)

    # Convert engine_args to dict for LLM initialization
    llm_kwargs = dataclasses.asdict(engine_args)

    logger.info("Initializing vLLM engine...")
    llm = LLM(**llm_kwargs)

    # Generate warmup requests
    logger.info(
        "Running initial warmup inference "
        "(batch_size=%d, input_len=%d, output_len=%d)...",
        args.batch_size,
        args.input_len,
        args.output_len,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    # Create dummy prompts for warmup
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts: list[PromptType] = [
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
    ]

    # Run warmup inference
    llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

    # Access model_runner directly for prefill benchmarking
    worker = llm.llm_engine.model_executor.driver_worker.worker

    # Run benchmarks
    logger.info("Starting prefill (eager mode) benchmarks...")
    start_time = time.perf_counter()

    all_results: dict[int, dict[str, Any]] = {}
    for num_tokens in args.prefill_sizes:
        logger.info("Benchmarking prefill for %d tokens...", num_tokens)
        result = worker.benchmark_prefill(
            num_tokens=num_tokens,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
        )
        all_results[num_tokens] = result

    end_time = time.perf_counter()

    # Print summary
    print("\n" + "=" * 60)
    print("Prefill (Eager Mode) Benchmark Results")
    print("=" * 60)

    for num_tokens, result in all_results.items():
        print(f"\nPrefill ({num_tokens} tokens):")
        print(f"  Mean:   {result['mean_ms']:.3f} ms")
        print(f"  Median: {result['median_ms']:.3f} ms")
        print(f"  Std:    {result['std_ms']:.3f} ms")
        print(f"  Min:    {result['min_ms']:.3f} ms")
        print(f"  Max:    {result['max_ms']:.3f} ms")
        print(f"  P50:    {result['percentiles']['p50']:.3f} ms")
        print(f"  P90:    {result['percentiles']['p90']:.3f} ms")
        print(f"  P95:    {result['percentiles']['p95']:.3f} ms")
        print(f"  P99:    {result['percentiles']['p99']:.3f} ms")

    print("\n" + "-" * 60)
    print(f"Total benchmark time: {end_time - start_time:.2f} seconds")
    print("=" * 60)

    # Save results if requested
    if args.output_json:
        output_data = {
            "args": {
                "model": args.model,
                "prefill_sizes": args.prefill_sizes,
                "batch_size": args.batch_size,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "num_iterations": args.num_iterations,
                "warmup_iterations": args.warmup_iterations,
            },
            "results": {str(k): v for k, v in all_results.items()},
            "total_time_seconds": end_time - start_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", args.output_json)
        save_to_pytorch_benchmark_format(args, all_results)
