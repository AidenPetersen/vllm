# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark captured CUDA/HIP graphs in vLLM.

This module provides utilities for benchmarking the performance of captured
CUDA graphs used by vLLM during inference. It allows re-executing graphs
multiple times to measure kernel performance accurately.
"""

import argparse
import dataclasses
import json
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class GraphBenchmarkResult:
    """Results from benchmarking a single CUDA graph."""

    graph_key: str
    """Identifier for the benchmarked graph."""

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
            "graph_key": self.graph_key,
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
    def from_dict(cls, data: dict[str, Any]) -> "GraphBenchmarkResult":
        """Create from dictionary."""
        return cls(**data)


class GraphBenchmarkRunner:
    """Runner for benchmarking captured CUDA/HIP graphs.

    This class provides methods to benchmark graphs captured during vLLM
    inference. It handles warmup, timing, and statistics collection.

    Example:
        >>> from vllm import LLM
        >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
        >>> # Warmup to trigger graph capture
        >>> llm.generate(["Hello"], max_tokens=1)
        >>> # Run benchmarks
        >>> runner = GraphBenchmarkRunner(llm)
        >>> results = runner.benchmark_all()
    """

    def __init__(self, llm: Any):
        """Initialize the benchmark runner.

        Args:
            llm: A vLLM LLM instance with captured CUDA graphs.
        """
        self.llm = llm

    def get_captured_graphs(self) -> dict[Any, torch.cuda.CUDAGraph]:
        """Get all captured CUDA graphs from the LLM engine.

        Returns:
            Dictionary mapping graph keys to CUDAGraph objects.
        """
        # Access graphs through the worker
        return self.llm.llm_engine.model_executor.driver_worker.model_runner.get_captured_graphs()

    def list_graphs(self) -> list[str]:
        """List all available captured graph keys.

        Returns:
            List of graph key strings.
        """
        graphs = self.get_captured_graphs()
        return [str(key) for key in graphs.keys()]

    @torch.inference_mode()
    def benchmark_graph(
        self,
        graph_key: Any | None = None,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> GraphBenchmarkResult:
        """Benchmark a specific captured CUDA graph.

        Args:
            graph_key: Key identifying the graph. None uses the first available.
            num_iterations: Number of timed iterations.
            warmup_iterations: Number of warmup iterations.

        Returns:
            GraphBenchmarkResult with timing statistics.

        Raises:
            ValueError: If no graphs are captured or key not found.
        """
        graphs = self.get_captured_graphs()

        if not graphs:
            raise ValueError(
                "No CUDA graphs captured. Ensure the model has been warmed up "
                "with actual inference requests."
            )

        if graph_key is None:
            graph_key = next(iter(graphs.keys()))
            logger.info("Using first available graph: %s", graph_key)

        if graph_key not in graphs:
            raise ValueError(
                f"Graph key {graph_key} not found. "
                f"Available: {list(graphs.keys())}"
            )

        graph = graphs[graph_key]

        # Create CUDA events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        logger.info("Running %d warmup iterations...", warmup_iterations)
        for _ in range(warmup_iterations):
            graph.replay()
        torch.cuda.synchronize()

        # Timed iterations
        logger.info("Running %d timed iterations...", num_iterations)
        times_ms: list[float] = []

        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        # Calculate statistics
        times_array = np.array(times_ms)
        percentiles = {
            "p50": float(np.percentile(times_array, 50)),
            "p90": float(np.percentile(times_array, 90)),
            "p95": float(np.percentile(times_array, 95)),
            "p99": float(np.percentile(times_array, 99)),
        }

        result = GraphBenchmarkResult(
            graph_key=str(graph_key),
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            mean_ms=float(np.mean(times_array)),
            median_ms=float(np.median(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            percentiles=percentiles,
            all_times_ms=times_ms,
        )

        logger.info(
            "Benchmark complete: mean=%.3fms, median=%.3fms, "
            "std=%.3fms, min=%.3fms, max=%.3fms",
            result.mean_ms,
            result.median_ms,
            result.std_ms,
            result.min_ms,
            result.max_ms,
        )

        return result

    def benchmark_all(
        self,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, GraphBenchmarkResult]:
        """Benchmark all captured CUDA graphs.

        Args:
            num_iterations: Number of timed iterations per graph.
            warmup_iterations: Number of warmup iterations per graph.

        Returns:
            Dictionary mapping graph key strings to results.
        """
        graphs = self.get_captured_graphs()

        if not graphs:
            raise ValueError("No CUDA graphs captured.")

        results: dict[str, GraphBenchmarkResult] = {}
        for graph_key in tqdm(list(graphs.keys()), desc="Graphs"):
            key_str = str(graph_key)
            logger.info("Benchmarking graph: %s", key_str)
            results[key_str] = self.benchmark_graph(
                graph_key=graph_key,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )

        return results


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    """Save results in PyTorch benchmark format."""
    metrics = {}
    for key, result in results.items():
        metrics[f"graph_{key}_mean_ms"] = [result["mean_ms"]]
        metrics[f"graph_{key}_median_ms"] = [result["median_ms"]]

    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics=metrics,
        extra_info={"results": results},
    )
    if pt_records:
        import os

        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for graph benchmarking."""
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of timed iterations per graph.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--graph-size",
        type=int,
        default=None,
        help="Specific graph size (num_tokens) to benchmark. "
        "If not specified, benchmarks all captured graphs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size to use for warmup requests that trigger graph capture.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=32,
        help="Input length for warmup requests.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=16,
        help="Output length for warmup requests.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON.",
    )
    parser.add_argument(
        "--list-graphs",
        action="store_true",
        help="List available captured graphs and exit.",
    )

    # Add engine args
    parser = EngineArgs.add_cli_args(parser)
    # Enable prefix caching disabled by default for benchmarking
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace) -> None:
    """Main function for graph benchmarking CLI."""
    # Lazy imports
    from vllm import LLM, SamplingParams

    engine_args = EngineArgs.from_cli_args(args)

    logger.info("Initializing vLLM engine...")
    llm = LLM(**dataclasses.asdict(engine_args))

    # Generate warmup requests to trigger graph capture
    logger.info(
        "Running warmup requests to trigger graph capture "
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

    # Run warmup to capture graphs
    llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

    # Create benchmark runner
    runner = GraphBenchmarkRunner(llm)

    # List graphs if requested
    if args.list_graphs:
        graphs = runner.list_graphs()
        print("\nCaptured CUDA graphs:")
        for i, key in enumerate(graphs, 1):
            print(f"  {i}. {key}")
        print(f"\nTotal: {len(graphs)} graphs")
        return

    # Run benchmarks
    logger.info("Starting graph benchmarks...")
    start_time = time.perf_counter()

    if args.graph_size is not None:
        # Benchmark specific graph size
        result = runner.benchmark_graph(
            graph_key=args.graph_size,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
        )
        all_results = {str(args.graph_size): result.to_dict()}
    else:
        # Benchmark all graphs
        results = runner.benchmark_all(
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
        )
        all_results = {key: res.to_dict() for key, res in results.items()}

    end_time = time.perf_counter()

    # Print summary
    print("\n" + "=" * 60)
    print("CUDA Graph Benchmark Results")
    print("=" * 60)

    for key, result in all_results.items():
        print(f"\nGraph: {key}")
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
                "batch_size": args.batch_size,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "num_iterations": args.num_iterations,
                "warmup_iterations": args.warmup_iterations,
            },
            "results": all_results,
            "total_time_seconds": end_time - start_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", args.output_json)
        save_to_pytorch_benchmark_format(args, all_results)
