#!/usr/bin/env python3
# LLKJJ ML Pipeline - Performance Benchmarking Suite

"""
Automated performance benchmarking and optimization for LLKJJ ML Pipeline.
Provides comprehensive performance testing with real-world scenarios.
"""

import json
import logging
import statistics
import time
import tracemalloc
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import psutil

# Import LLKJJ ML components
from llkjj_ml_plugin import process_pdf_simple


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""

    test_name: str
    operation_type: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    success: bool
    error_message: str | None = None
    throughput_ops_per_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        return data


@dataclass
class PerformanceProfile:
    """Performance profile summary."""

    test_suite: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    max_memory_mb: float
    avg_cpu_percent: float
    total_throughput_ops_per_sec: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PerformanceBenchmark:
    """Advanced performance benchmarking suite."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.results: list[BenchmarkResult] = []

        # Test data
        self.test_pdf = Path("test_pdfs/Sonepar_test3.pdf")

    def measure_operation(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Measure a single operation's performance."""
        # Start monitoring
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        start_time = datetime.now()

        success = True
        error_message = None

        try:
            # Execute operation
            operation(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)

        # End monitoring
        end_time = datetime.now()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration_ms = (end_time - start_time).total_seconds() * 1000
        memory_peak_mb = peak / 1024 / 1024
        memory_delta_mb = end_memory - start_memory
        avg_cpu = (start_cpu + end_cpu) / 2

        benchmark_result = BenchmarkResult(
            test_name=operation_name,
            operation_type="single",
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            memory_peak_mb=memory_peak_mb,
            memory_delta_mb=memory_delta_mb,
            cpu_percent=avg_cpu,
            success=success,
            error_message=error_message,
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def benchmark_pdf_processing(
        self, num_iterations: int = 10
    ) -> list[BenchmarkResult]:
        """Benchmark PDF processing operations."""
        results = []

        self.logger.info(
            f"Starting PDF processing benchmark ({num_iterations} iterations)"
        )

        for i in range(num_iterations):
            self.logger.info(f"PDF processing iteration {i+1}/{num_iterations}")

            result = self.measure_operation(
                process_pdf_simple, f"pdf_processing_iter_{i+1}", str(self.test_pdf)
            )
            results.append(result)

            # Small delay between iterations
            time.sleep(0.1)

        return results

    def benchmark_concurrent_processing(
        self, num_threads: int = 4, operations_per_thread: int = 5
    ) -> list[BenchmarkResult]:
        """Benchmark concurrent processing with multiple threads."""
        results = []
        start_time = datetime.now()

        self.logger.info(
            f"Starting concurrent benchmark ({num_threads} threads, {operations_per_thread} ops/thread)"
        )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = []
            for thread_id in range(num_threads):
                for op_id in range(operations_per_thread):
                    future = executor.submit(
                        self._concurrent_operation, f"concurrent_t{thread_id}_op{op_id}"
                    )
                    futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Concurrent operation failed: {e}")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        total_ops = num_threads * operations_per_thread
        throughput = total_ops / total_duration

        self.logger.info(f"Concurrent benchmark completed: {throughput:.2f} ops/sec")

        # Update throughput for all results
        for result in results:
            result.throughput_ops_per_sec = throughput

        return results

    def _concurrent_operation(self, operation_name: str) -> BenchmarkResult:
        """Single concurrent operation for threading benchmark."""
        return self.measure_operation(
            process_pdf_simple, operation_name, str(self.test_pdf)
        )

    def benchmark_memory_stress(
        self, num_iterations: int = 20
    ) -> list[BenchmarkResult]:
        """Benchmark memory usage under stress conditions."""
        results = []

        self.logger.info(f"Starting memory stress test ({num_iterations} iterations)")

        # Rapid successive operations to test memory management
        for i in range(num_iterations):
            result = self.measure_operation(
                process_pdf_simple, f"memory_stress_iter_{i+1}", str(self.test_pdf)
            )
            results.append(result)

            # No delay to stress memory
            if i % 5 == 0:
                self.logger.info(f"Memory stress iteration {i+1}/{num_iterations}")

        return results

    def run_comprehensive_benchmark(self) -> PerformanceProfile:
        """Run comprehensive performance benchmark suite."""
        self.logger.info("🚀 Starting comprehensive performance benchmark")
        self.results.clear()

        # Individual operations
        self.logger.info("📊 Phase 1: Individual PDF processing")
        pdf_results = self.benchmark_pdf_processing(10)

        # Concurrent operations
        self.logger.info("📊 Phase 2: Concurrent processing")
        concurrent_results = self.benchmark_concurrent_processing(4, 5)

        # Memory stress test
        self.logger.info("📊 Phase 3: Memory stress test")
        memory_results = self.benchmark_memory_stress(20)

        # Analyze results
        all_results = pdf_results + concurrent_results + memory_results
        profile = self._analyze_results(all_results, "comprehensive_benchmark")

        # Save results
        self._save_results(all_results, profile)

        self.logger.info("✅ Comprehensive benchmark completed")
        return profile

    def _analyze_results(
        self, results: list[BenchmarkResult], suite_name: str
    ) -> PerformanceProfile:
        """Analyze benchmark results and create performance profile."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            raise ValueError("No successful benchmark results to analyze")

        durations = [r.duration_ms for r in successful_results]
        memory_usage = [r.memory_peak_mb for r in successful_results]
        cpu_usage = [r.cpu_percent for r in successful_results]

        # Calculate throughput
        total_duration_sec = sum(durations) / 1000
        throughput = (
            len(successful_results) / total_duration_sec
            if total_duration_sec > 0
            else 0
        )

        profile = PerformanceProfile(
            test_suite=suite_name,
            total_tests=len(results),
            successful_tests=len(successful_results),
            failed_tests=len(results) - len(successful_results),
            avg_duration_ms=statistics.mean(durations),
            p50_duration_ms=statistics.median(durations),
            p95_duration_ms=(
                statistics.quantiles(durations, n=20)[18]
                if len(durations) >= 20
                else max(durations)
            ),
            p99_duration_ms=(
                statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else max(durations)
            ),
            max_memory_mb=max(memory_usage),
            avg_cpu_percent=statistics.mean(cpu_usage),
            total_throughput_ops_per_sec=throughput,
            timestamp=datetime.now(),
        )

        return profile

    def _save_results(
        self, results: list[BenchmarkResult], profile: PerformanceProfile
    ) -> None:
        """Save benchmark results and create visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        # Save profile
        profile_file = self.output_dir / f"performance_profile_{timestamp}.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)

        # Create visualizations
        self._create_visualizations(results, timestamp)

        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Profile saved to {profile_file}")

    def _create_visualizations(
        self, results: list[BenchmarkResult], timestamp: str
    ) -> None:
        """Create performance visualization charts."""
        if not results:
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([r.to_dict() for r in results])

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Performance Benchmark Results - {timestamp}", fontsize=16)

        # Duration distribution
        axes[0, 0].hist(
            df["duration_ms"], bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Duration Distribution (ms)")
        axes[0, 0].set_xlabel("Duration (ms)")
        axes[0, 0].set_ylabel("Frequency")

        # Memory usage over time
        axes[0, 1].plot(df.index, df["memory_peak_mb"], marker="o", color="orange")
        axes[0, 1].set_title("Memory Usage Over Time")
        axes[0, 1].set_xlabel("Test Number")
        axes[0, 1].set_ylabel("Peak Memory (MB)")

        # CPU usage over time
        axes[1, 0].plot(df.index, df["cpu_percent"], marker="s", color="green")
        axes[1, 0].set_title("CPU Usage Over Time")
        axes[1, 0].set_xlabel("Test Number")
        axes[1, 0].set_ylabel("CPU Usage (%)")

        # Success rate
        success_counts = df["success"].value_counts()
        axes[1, 1].pie(
            success_counts.values,
            labels=["Success", "Failed"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
        )
        axes[1, 1].set_title("Success Rate")

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / f"performance_charts_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Performance charts saved to {plot_file}")

    def compare_with_baseline(
        self, baseline_file: Path | None = None
    ) -> dict[str, Any]:
        """Compare current results with baseline performance."""
        if not baseline_file:
            # Find most recent baseline
            baseline_files = list(self.output_dir.glob("performance_profile_*.json"))
            if not baseline_files:
                self.logger.warning("No baseline found for comparison")
                return {}
            baseline_file = max(baseline_files, key=lambda x: x.stat().st_mtime)

        with open(baseline_file, encoding="utf-8") as f:
            baseline = json.load(f)

        # Run current benchmark
        current_profile = self.run_comprehensive_benchmark()

        # Compare metrics
        comparison = {
            "baseline_file": str(baseline_file),
            "baseline_timestamp": baseline["timestamp"],
            "current_timestamp": current_profile.timestamp.isoformat(),
            "improvements": {},
            "regressions": {},
        }

        metrics = [
            "avg_duration_ms",
            "p95_duration_ms",
            "max_memory_mb",
            "avg_cpu_percent",
            "total_throughput_ops_per_sec",
        ]

        for metric in metrics:
            baseline_value = baseline.get(metric, 0)
            current_value = getattr(current_profile, metric)

            if baseline_value > 0:
                change_percent = (
                    (current_value - baseline_value) / baseline_value
                ) * 100

                if metric == "total_throughput_ops_per_sec":
                    # Higher is better for throughput
                    if change_percent > 0:
                        comparison["improvements"][metric] = change_percent
                    else:
                        comparison["regressions"][metric] = abs(change_percent)
                else:
                    # Lower is better for other metrics
                    if change_percent < 0:
                        comparison["improvements"][metric] = abs(change_percent)
                    else:
                        comparison["regressions"][metric] = change_percent

        return comparison


def main() -> None:
    """Run performance benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="LLKJJ ML Performance Benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument("--baseline", type=Path, help="Baseline file for comparison")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for basic tests",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of threads for concurrent tests"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    )

    # Create benchmark suite
    benchmark = PerformanceBenchmark(args.output)

    print("🚀 LLKJJ ML Performance Benchmark Suite")
    print(f"Output directory: {args.output}")
    print(f"Iterations: {args.iterations}")
    print(f"Concurrent threads: {args.threads}")

    try:
        if args.baseline:
            print(f"Comparing with baseline: {args.baseline}")
            comparison = benchmark.compare_with_baseline(args.baseline)
            print("\n📊 Performance Comparison:")
            print(json.dumps(comparison, indent=2))
        else:
            print("\n🏃 Running comprehensive benchmark...")
            profile = benchmark.run_comprehensive_benchmark()

            print("\n📈 Performance Profile:")
            print(f"Total tests: {profile.total_tests}")
            print(
                f"Success rate: {profile.successful_tests}/{profile.total_tests} ({profile.successful_tests/profile.total_tests*100:.1f}%)"
            )
            print(f"Average duration: {profile.avg_duration_ms:.2f}ms")
            print(f"P95 duration: {profile.p95_duration_ms:.2f}ms")
            print(f"Peak memory: {profile.max_memory_mb:.2f}MB")
            print(f"Throughput: {profile.total_throughput_ops_per_sec:.2f} ops/sec")

        print(f"\n✅ Benchmark completed! Results saved to {args.output}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
