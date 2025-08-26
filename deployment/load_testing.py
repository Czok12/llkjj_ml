#!/usr/bin/env python3
# LLKJJ ML Pipeline - Load Testing Suite

"""
Load testing for LLKJJ ML Pipeline under realistic production conditions.
Simulates high-volume document processing scenarios.
"""

import json
import logging
import queue
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from llkjj_ml_plugin import process_pdf_simple

from deployment.monitoring import get_performance_monitor, get_production_logger


@dataclass
class LoadTestConfig:
    """Load test configuration."""

    test_name: str
    duration_minutes: int
    concurrent_users: int
    ramp_up_minutes: int
    operations_per_user: int
    think_time_min_sec: float
    think_time_max_sec: float
    target_throughput_ops_per_sec: float | None = None


@dataclass
class LoadTestResult:
    """Load test result."""

    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    peak_memory_mb: float
    avg_cpu_percent: float
    errors_by_type: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["config"] = asdict(self.config)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        return data


@dataclass
class OperationResult:
    """Single operation result."""

    user_id: int
    operation_id: int
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error_type: str | None = None
    error_message: str | None = None


class LoadTestUser:
    """Simulates a single user performing operations."""

    def __init__(
        self,
        user_id: int,
        config: LoadTestConfig,
        results_queue: queue.Queue[OperationResult],
    ):
        self.user_id = user_id
        self.config = config
        self.results_queue = results_queue
        self.logger = logging.getLogger(f"user_{user_id}")
        self.test_pdf = Path("test_pdfs/Sonepar_test3.pdf")

    def run_operations(self, stop_event: threading.Event) -> None:
        """Run operations for this user until stopped."""
        operation_id = 0

        while (
            not stop_event.is_set() and operation_id < self.config.operations_per_user
        ):
            operation_id += 1

            # Perform operation
            start_time = datetime.now()
            success = True
            error_type = None
            error_message = None

            try:
                result = process_pdf_simple(str(self.test_pdf))
                if not result:
                    success = False
                    error_type = "EmptyResult"
                    error_message = "Processing returned empty result"
            except Exception as e:
                success = False
                error_type = type(e).__name__
                error_message = str(e)

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Record result
            op_result = OperationResult(
                user_id=self.user_id,
                operation_id=operation_id,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
                error_message=error_message,
            )

            self.results_queue.put(op_result)

            # Think time (simulate user delay)
            if not stop_event.is_set():
                think_time = random.uniform(
                    self.config.think_time_min_sec, self.config.think_time_max_sec
                )
                time.sleep(think_time)


class LoadTester:
    """Advanced load testing framework."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("load_test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = get_production_logger()
        self.performance_monitor = get_performance_monitor()

    def create_test_scenarios(self) -> list[LoadTestConfig]:
        """Create predefined load test scenarios."""
        scenarios = [
            # Light load - typical usage
            LoadTestConfig(
                test_name="light_load",
                duration_minutes=5,
                concurrent_users=2,
                ramp_up_minutes=1,
                operations_per_user=20,
                think_time_min_sec=1.0,
                think_time_max_sec=3.0,
                target_throughput_ops_per_sec=1.0,
            ),
            # Medium load - busy periods
            LoadTestConfig(
                test_name="medium_load",
                duration_minutes=10,
                concurrent_users=5,
                ramp_up_minutes=2,
                operations_per_user=30,
                think_time_min_sec=0.5,
                think_time_max_sec=2.0,
                target_throughput_ops_per_sec=3.0,
            ),
            # Heavy load - peak usage
            LoadTestConfig(
                test_name="heavy_load",
                duration_minutes=15,
                concurrent_users=10,
                ramp_up_minutes=3,
                operations_per_user=50,
                think_time_min_sec=0.1,
                think_time_max_sec=1.0,
                target_throughput_ops_per_sec=8.0,
            ),
            # Stress test - beyond normal capacity
            LoadTestConfig(
                test_name="stress_test",
                duration_minutes=20,
                concurrent_users=20,
                ramp_up_minutes=5,
                operations_per_user=100,
                think_time_min_sec=0.05,
                think_time_max_sec=0.5,
                target_throughput_ops_per_sec=15.0,
            ),
        ]

        return scenarios

    def run_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Execute a single load test."""
        logging.getLogger(__name__).info(f"🚀 Starting load test: {config.test_name}")
        logging.getLogger(__name__).info(
            f"Config: {config.concurrent_users} users, {config.duration_minutes}min"
        )

        # Results collection
        results_queue: queue.Queue[OperationResult] = queue.Queue()
        operation_results: list[OperationResult] = []

        # System monitoring
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = datetime.now()

        # Thread management
        stop_event = threading.Event()
        user_threads = []

        try:
            # Start users with ramp-up
            ramp_up_delay = (config.ramp_up_minutes * 60) / config.concurrent_users

            for user_id in range(config.concurrent_users):
                user = LoadTestUser(user_id, config, results_queue)
                thread = threading.Thread(
                    target=user.run_operations,
                    args=(stop_event,),
                    name=f"user_{user_id}",
                )
                user_threads.append(thread)
                thread.start()

                # Ramp-up delay
                if user_id < config.concurrent_users - 1:
                    time.sleep(ramp_up_delay)

            # Monitor for duration
            test_duration = config.duration_minutes * 60
            start_monitor = time.time()
            cpu_samples = []

            while time.time() - start_monitor < test_duration:
                # Collect CPU samples
                cpu_samples.append(psutil.cpu_percent())

                # Collect operation results
                while not results_queue.empty():
                    try:
                        result = results_queue.get_nowait()
                        operation_results.append(result)
                    except queue.Empty:
                        break

                time.sleep(1)  # Sample every second

            # Stop all users
            stop_event.set()

            # Wait for threads to finish
            for thread in user_threads:
                thread.join(timeout=30)

            # Collect remaining results
            while not results_queue.empty():
                try:
                    result = results_queue.get_nowait()
                    operation_results.append(result)
                except queue.Empty:
                    break

        except Exception as e:
            logging.getLogger(__name__).error(f"Load test failed: {e}")
            stop_event.set()
            raise

        # Calculate metrics
        end_time = datetime.now()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        test_result = self._analyze_load_test_results(
            config,
            start_time,
            end_time,
            operation_results,
            end_memory - start_memory,
            statistics.mean(cpu_samples) if cpu_samples else 0,
        )

        # Save results
        self._save_load_test_results(test_result, operation_results)

        logging.getLogger(__name__).info(f"✅ Load test completed: {config.test_name}")
        logging.getLogger(__name__).info(
            f"Results: {test_result.successful_operations}/{test_result.total_operations} success, "
            f"{test_result.throughput_ops_per_sec:.2f} ops/sec"
        )

        return test_result

    def _analyze_load_test_results(
        self,
        config: LoadTestConfig,
        start_time: datetime,
        end_time: datetime,
        operation_results: list[OperationResult],
        memory_delta_mb: float,
        avg_cpu: float,
    ) -> LoadTestResult:
        """Analyze load test results and calculate metrics."""
        if not operation_results:
            raise ValueError("No operation results to analyze")

        successful_ops = [r for r in operation_results if r.success]
        failed_ops = [r for r in operation_results if not r.success]

        # Response time analysis
        if successful_ops:
            response_times = [r.duration_ms for r in successful_ops]
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)

            # Calculate percentiles
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            p95_index = int(0.95 * n)
            p99_index = int(0.99 * n)

            p95_response_time = sorted_times[min(p95_index, n - 1)]
            p99_response_time = sorted_times[min(p99_index, n - 1)]
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = p50_response_time = p95_response_time = 0
            p99_response_time = max_response_time = min_response_time = 0

        # Throughput calculation
        test_duration_sec = (end_time - start_time).total_seconds()
        throughput = (
            len(successful_ops) / test_duration_sec if test_duration_sec > 0 else 0
        )

        # Error analysis
        error_rate = (
            (len(failed_ops) / len(operation_results)) * 100 if operation_results else 0
        )
        errors_by_type: dict[str, int] = {}
        for op in failed_ops:
            error_type = op.error_type or "Unknown"
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        return LoadTestResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_operations=len(operation_results),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_response_time_ms=max_response_time,
            min_response_time_ms=min_response_time,
            throughput_ops_per_sec=throughput,
            error_rate_percent=error_rate,
            peak_memory_mb=memory_delta_mb,
            avg_cpu_percent=avg_cpu,
            errors_by_type=errors_by_type,
        )

    def _save_load_test_results(
        self, test_result: LoadTestResult, operation_results: list[OperationResult]
    ) -> None:
        """Save load test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = test_result.config.test_name

        # Save test result summary
        summary_file = self.output_dir / f"load_test_{test_name}_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(test_result.to_dict(), f, indent=2)

        # Save detailed operation results
        details_file = self.output_dir / f"operations_{test_name}_{timestamp}.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump([asdict(op) for op in operation_results], f, indent=2)

        logging.getLogger(__name__).info(f"Load test results saved to {summary_file}")

    def run_all_scenarios(self) -> list[LoadTestResult]:
        """Run all predefined load test scenarios."""
        scenarios = self.create_test_scenarios()
        results = []

        logging.getLogger(__name__).info(
            f"🚀 Starting load test suite with {len(scenarios)} scenarios"
        )

        for i, scenario in enumerate(scenarios, 1):
            logging.getLogger(__name__).info(
                f"Running scenario {i}/{len(scenarios)}: {scenario.test_name}"
            )

            try:
                result = self.run_load_test(scenario)
                results.append(result)

                # Brief pause between tests
                time.sleep(10)

            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Scenario {scenario.test_name} failed: {e}"
                )
                continue

        # Generate summary report
        self._generate_summary_report(results)

        logging.getLogger(__name__).info("✅ Load test suite completed")
        return results

    def _generate_summary_report(self, results: list[LoadTestResult]) -> None:
        """Generate a summary report of all load tests."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"load_test_summary_{timestamp}.md"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# LLKJJ ML Pipeline - Load Test Summary Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            f.write("## Test Results Summary\n\n")
            f.write(
                "| Test Name | Users | Duration | Success Rate | Throughput (ops/sec) | Avg Response (ms) | P95 Response (ms) |\n"
            )
            f.write(
                "|-----------|-------|----------|--------------|---------------------|------------------|------------------|\n"
            )

            for result in results:
                success_rate = (
                    result.successful_operations / result.total_operations
                ) * 100
                f.write(
                    f"| {result.config.test_name} | {result.config.concurrent_users} | "
                    f"{result.config.duration_minutes}min | {success_rate:.1f}% | "
                    f"{result.throughput_ops_per_sec:.2f} | {result.avg_response_time_ms:.2f} | "
                    f"{result.p95_response_time_ms:.2f} |\n"
                )

            f.write("\n## Detailed Analysis\n\n")

            for result in results:
                f.write(f"### {result.config.test_name}\n\n")
                f.write(
                    f"- **Configuration**: {result.config.concurrent_users} concurrent users, "
                    f"{result.config.duration_minutes} minutes\n"
                )
                f.write(f"- **Total Operations**: {result.total_operations}\n")
                f.write(
                    f"- **Success Rate**: {(result.successful_operations/result.total_operations)*100:.1f}%\n"
                )
                f.write(
                    f"- **Throughput**: {result.throughput_ops_per_sec:.2f} ops/sec\n"
                )
                f.write(
                    f"- **Response Times**: Avg {result.avg_response_time_ms:.2f}ms, "
                    f"P95 {result.p95_response_time_ms:.2f}ms, P99 {result.p99_response_time_ms:.2f}ms\n"
                )
                f.write(
                    f"- **Resource Usage**: {result.peak_memory_mb:.2f}MB memory, "
                    f"{result.avg_cpu_percent:.1f}% CPU\n"
                )

                if result.errors_by_type:
                    f.write(f"- **Errors**: {result.errors_by_type}\n")

                f.write("\n")

        logging.getLogger(__name__).info(f"Summary report saved to {report_file}")


def main() -> None:
    """Run load testing."""
    import argparse

    parser = argparse.ArgumentParser(description="LLKJJ ML Load Testing")
    parser.add_argument(
        "--output",
        type=Path,
        default="load_test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--scenario",
        choices=["light", "medium", "heavy", "stress", "all"],
        default="all",
        help="Test scenario to run",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    )

    # Create load tester
    load_tester = LoadTester(args.output)

    print("🚀 LLKJJ ML Load Testing Suite")
    print(f"Output directory: {args.output}")
    print(f"Scenario: {args.scenario}")

    try:
        if args.scenario == "all":
            print("\n🏃 Running all load test scenarios...")
            results = load_tester.run_all_scenarios()

            print("\n📊 Load Test Suite Results:")
            for result in results:
                success_rate = (
                    result.successful_operations / result.total_operations
                ) * 100
                print(
                    f"  {result.config.test_name}: {success_rate:.1f}% success, "
                    f"{result.throughput_ops_per_sec:.2f} ops/sec"
                )
        else:
            # Run single scenario
            scenarios = load_tester.create_test_scenarios()
            scenario_map = {
                s.test_name.replace("_load", "").replace("_test", ""): s
                for s in scenarios
            }

            if args.scenario not in scenario_map:
                print(f"❌ Unknown scenario: {args.scenario}")
                print(f"Available scenarios: {list(scenario_map.keys())}")
                return

            config = scenario_map[args.scenario]
            print(f"\n🏃 Running {config.test_name}...")
            result = load_tester.run_load_test(config)

            success_rate = (
                result.successful_operations / result.total_operations
            ) * 100
            print("\n📊 Load Test Result:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
            print(f"  Avg response time: {result.avg_response_time_ms:.2f}ms")
            print(f"  P95 response time: {result.p95_response_time_ms:.2f}ms")

        print(f"\n✅ Load testing completed! Results saved to {args.output}")

    except Exception as e:
        print(f"❌ Load testing failed: {e}")
        raise


if __name__ == "__main__":
    main()
