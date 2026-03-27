# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark CUDA/HIP graph replay performance.

This module provides utilities for benchmarking the performance of captured
CUDA graphs used by vLLM. It allows re-executing graphs multiple times to
measure kernel performance accurately.

Output is JSON written to a file specified by -o/--output:
{
    "batch_size_1": {"time": <geomean_ms>, "weight": <1/num_graphs>},
    "batch_size_8": {"time": <geomean_ms>, "weight": <1/num_graphs>},
    ...
}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from vllm import LLM


@dataclasses.dataclass
class GraphBenchmarkResult:
    """Results from benchmarking a single CUDA graph."""

    graph_key: str
    """Identifier for the benchmarked graph."""

    num_iterations: int
    """Number of timed iterations run."""

    warmup_iterations: int
    """Number of warmup iterations run before timing."""

    geomean_ms: float
    """Geometric mean execution time in milliseconds."""

    mean_ms: float
    """Arithmetic mean execution time in milliseconds."""

    median_ms: float
    """Median execution time in milliseconds."""

    std_ms: float
    """Standard deviation of execution time in milliseconds."""

    min_ms: float
    """Minimum execution time in milliseconds."""

    max_ms: float
    """Maximum execution time in milliseconds."""

    all_times_ms: list[float]
    """List of all measured execution times."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "graph_key": self.graph_key,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "geomean_ms": self.geomean_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "all_times_ms": self.all_times_ms,
        }


class GraphBenchmarkRunner:
    """Runner for benchmarking captured CUDA/HIP graphs.

    This class provides methods to benchmark graphs captured during vLLM
    inference. It handles warmup, timing, and statistics collection.
    """

    def __init__(self, llm: LLM):
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
        # Access graphs through the worker.
        # driver_worker is a WorkerWrapperBase, which contains the actual
        # Worker instance in its 'worker' attribute.
        return self.llm.llm_engine.model_executor.driver_worker.worker.get_captured_graphs()

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
        from vllm.logger import init_logger

        logger = init_logger(__name__)

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
        # Geometric mean: exp(mean(log(x)))
        geomean = float(np.exp(np.mean(np.log(times_array))))

        result = GraphBenchmarkResult(
            graph_key=str(graph_key),
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            geomean_ms=geomean,
            mean_ms=float(np.mean(times_array)),
            median_ms=float(np.median(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            all_times_ms=times_ms,
        )

        logger.info(
            "Benchmark complete: geomean=%.3fms, mean=%.3fms, median=%.3fms, "
            "std=%.3fms, min=%.3fms, max=%.3fms",
            result.geomean_ms,
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
        from vllm.logger import init_logger

        logger = init_logger(__name__)

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


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for graph benchmarking."""
    # Import lazily to avoid importing vllm at module load time
    from vllm.engine.arg_utils import EngineArgs

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save benchmark results as JSON file (required).",
    )
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
        "--cudagraph-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Specific CUDA graph sizes (num_tokens) to capture. "
        "Example: --cudagraph-sizes 1 8 16 32. "
        "If not specified, uses vLLM's default sizes (1, 2, 4, 8, 16, ...).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to send for warmup inference after graph capture. "
        "This does NOT control which graph sizes are captured (use --cudagraph-sizes).",
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
    # Disable V1 multiprocessing to allow direct access to model components.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import PromptType
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    logger.info(
        "Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 for direct graph access."
    )

    engine_args = EngineArgs.from_cli_args(args)

    # Convert engine_args to dict for LLM initialization
    llm_kwargs = dataclasses.asdict(engine_args)

    # Handle cudagraph_sizes override
    if args.cudagraph_sizes is not None:
        cudagraph_sizes = sorted(args.cudagraph_sizes)
        logger.info(
            "Using custom cudagraph_capture_sizes: %s",
            cudagraph_sizes,
        )
        # Create or update compilation_config with custom sizes
        if llm_kwargs.get("compilation_config") is None:
            llm_kwargs["compilation_config"] = CompilationConfig(
                cudagraph_capture_sizes=cudagraph_sizes,
            )
        elif isinstance(llm_kwargs["compilation_config"], dict):
            llm_kwargs["compilation_config"]["cudagraph_capture_sizes"] = cudagraph_sizes
        elif isinstance(llm_kwargs["compilation_config"], CompilationConfig):
            llm_kwargs["compilation_config"].cudagraph_capture_sizes = cudagraph_sizes

    logger.info("Initializing vLLM engine...")
    llm = LLM(**llm_kwargs)

    # Generate warmup requests to exercise the captured graphs
    logger.info(
        "Running warmup inference "
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
    logger.info("Starting CUDA graph benchmarks...")

    if args.graph_size is not None:
        # Benchmark specific graph size
        result = runner.benchmark_graph(
            graph_key=args.graph_size,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
        )
        all_results = {str(args.graph_size): result}
    else:
        # Benchmark all graphs
        all_results = runner.benchmark_all(
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
        )

    # Build output JSON in the requested format:
    # { "batch_size_x": {"time": <geomean>, "weight": <1/num_graphs>} }
    num_graphs = len(all_results)
    weight = 1.0 / num_graphs if num_graphs > 0 else 0.0

    output_json: dict[str, dict[str, float]] = {}
    for key, result in all_results.items():
        output_json[f"batch_size_{key}"] = {
            "time": result.geomean_ms,
            "weight": weight,
        }

    # Write JSON to output file
    with open(args.output, "w") as f:
        json.dump(output_json, f, indent=2)

    logger.info("Results saved to %s", args.output)
