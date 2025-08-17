#!/usr/bin/env python3
# LLKJJ ML Pipeline - Complete Production Setup

"""
Complete production setup and management for LLKJJ ML Pipeline.
Integrates all deployment, monitoring, and testing components.
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from deployment.environment import initialize_environment
from deployment.load_testing import LoadTester
from deployment.monitoring import get_health_checker, get_production_logger
from deployment.performance_benchmark import PerformanceBenchmark


class ProductionManager:
    """Complete production management system."""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.production_logger = get_production_logger()
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent

        # Initialize environment
        self.config = initialize_environment(environment)

    def setup_environment(self) -> bool:
        """Setup and validate production environment."""
        self.logger.info(f"🚀 Setting up {self.environment} environment")

        try:
            # Check Poetry installation
            subprocess.run(["poetry", "--version"], check=True, capture_output=True)
            self.logger.info("✅ Poetry installation verified")

            # Install dependencies
            self.logger.info("📦 Installing dependencies...")
            subprocess.run(
                [
                    "poetry",
                    "install",
                    "--no-dev" if self.environment == "production" else "",
                ],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.logger.info("✅ Dependencies installed successfully")

            # Run health check
            if self._run_health_check():
                self.logger.info("✅ Environment setup completed successfully")
                return True
            else:
                self.logger.error("❌ Health check failed")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Environment setup failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during setup: {e}")
            return False

    def _run_health_check(self) -> bool:
        """Run comprehensive health check."""
        try:
            health_checker = get_health_checker()
            health_status = health_checker.check_system_health()

            if health_status.status == "healthy":
                self.logger.info("✅ System health check passed")
                return True
            else:
                self.logger.warning(f"⚠️ System health: {health_status.status}")
                for component, status in health_status.components.items():
                    if status != "healthy":
                        self.logger.warning(f"  {component}: {status}")
                return health_status.status != "unhealthy"
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return False

    def run_performance_tests(self) -> bool:
        """Run comprehensive performance testing."""
        self.logger.info("🏃 Running performance tests...")

        try:
            benchmark = PerformanceBenchmark()
            profile = benchmark.run_comprehensive_benchmark()

            # Check performance criteria
            success_criteria = {
                "success_rate": 0.95,  # 95% success rate
                "avg_duration_ms": 30000,  # 30 seconds max average
                "p95_duration_ms": 60000,  # 60 seconds P95
                "max_memory_mb": 2000,  # 2GB memory max
                "throughput_ops_per_sec": 0.1,  # Minimum throughput
            }

            success_rate = profile.successful_tests / profile.total_tests

            # Evaluate criteria
            passed = True
            if success_rate < success_criteria["success_rate"]:
                self.logger.error(f"❌ Success rate too low: {success_rate:.2%}")
                passed = False

            if profile.avg_duration_ms > success_criteria["avg_duration_ms"]:
                self.logger.error(
                    f"❌ Average duration too high: {profile.avg_duration_ms:.2f}ms"
                )
                passed = False

            if profile.p95_duration_ms > success_criteria["p95_duration_ms"]:
                self.logger.error(
                    f"❌ P95 duration too high: {profile.p95_duration_ms:.2f}ms"
                )
                passed = False

            if profile.max_memory_mb > success_criteria["max_memory_mb"]:
                self.logger.error(
                    f"❌ Memory usage too high: {profile.max_memory_mb:.2f}MB"
                )
                passed = False

            if (
                profile.total_throughput_ops_per_sec
                < success_criteria["throughput_ops_per_sec"]
            ):
                self.logger.error(
                    f"❌ Throughput too low: {profile.total_throughput_ops_per_sec:.2f} ops/sec"
                )
                passed = False

            if passed:
                self.logger.info("✅ Performance tests passed")
                self.logger.info(f"  Success rate: {success_rate:.2%}")
                self.logger.info(f"  Avg duration: {profile.avg_duration_ms:.2f}ms")
                self.logger.info(
                    f"  Throughput: {profile.total_throughput_ops_per_sec:.2f} ops/sec"
                )
            else:
                self.logger.error("❌ Performance tests failed")

            return passed

        except Exception as e:
            self.logger.error(f"❌ Performance testing failed: {e}")
            return False

    def run_load_tests(self, scenario: str = "medium") -> bool:
        """Run load testing scenarios."""
        self.logger.info(f"📊 Running load test scenario: {scenario}")

        try:
            load_tester = LoadTester()
            scenarios = load_tester.create_test_scenarios()

            # Find scenario
            scenario_config = None
            for config in scenarios:
                if scenario in config.test_name:
                    scenario_config = config
                    break

            if not scenario_config:
                self.logger.error(f"❌ Unknown scenario: {scenario}")
                return False

            result = load_tester.run_load_test(scenario_config)

            # Evaluate results
            success_rate = result.successful_operations / result.total_operations
            acceptable_success_rate = 0.90  # 90% for load tests

            if success_rate >= acceptable_success_rate:
                self.logger.info(
                    f"✅ Load test passed: {success_rate:.2%} success rate"
                )
                self.logger.info(
                    f"  Throughput: {result.throughput_ops_per_sec:.2f} ops/sec"
                )
                self.logger.info(
                    f"  P95 response time: {result.p95_response_time_ms:.2f}ms"
                )
                return True
            else:
                self.logger.error(
                    f"❌ Load test failed: {success_rate:.2%} success rate"
                )
                return False

        except Exception as e:
            self.logger.error(f"❌ Load testing failed: {e}")
            return False

    def run_security_audit(self) -> bool:
        """Run security audit."""
        self.logger.info("🔒 Running security audit...")

        try:
            # Run bandit security scan
            result = subprocess.run(
                [
                    "poetry",
                    "run",
                    "bandit",
                    "-r",
                    ".",
                    "-f",
                    "json",
                    "-o",
                    "security_report.json",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.logger.info("✅ Security audit passed - no issues found")
                return True
            else:
                self.logger.warning(
                    "⚠️ Security audit found issues - check security_report.json"
                )
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Security audit failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during security audit: {e}")
            return False

    def deploy_production(self) -> bool:
        """Deploy to production environment."""
        self.logger.info("🚀 Starting production deployment...")

        # Pre-deployment checks
        checks = [
            ("Environment Setup", self.setup_environment),
            ("Security Audit", self.run_security_audit),
            ("Performance Tests", self.run_performance_tests),
            ("Load Tests", lambda: self.run_load_tests("medium")),
        ]

        for check_name, check_func in checks:
            self.logger.info(f"Running {check_name}...")
            if not check_func():
                self.logger.error(f"❌ {check_name} failed - deployment aborted")
                return False
            time.sleep(2)  # Brief pause between checks

        # Deployment successful
        self.logger.info("✅ Production deployment completed successfully!")
        self.logger.info("🎉 LLKJJ ML Pipeline is production-ready!")

        return True

    def start_health_service(self, port: int = 8080) -> subprocess.Popen[bytes] | None:
        """Start health monitoring service."""
        self.logger.info(f"🩺 Starting health service on port {port}")

        try:
            process = subprocess.Popen(
                [
                    "poetry",
                    "run",
                    "python",
                    "deployment/health_service.py",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--env",
                    self.environment,
                ],
                cwd=self.project_root,
            )

            # Give it time to start
            time.sleep(3)

            # Check if process is still running
            if process.poll() is None:
                self.logger.info(f"✅ Health service started (PID: {process.pid})")
                self.logger.info(f"Health endpoint: http://localhost:{port}/health")
                self.logger.info(f"Metrics endpoint: http://localhost:{port}/metrics")
                return process
            else:
                self.logger.error("❌ Health service failed to start")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to start health service: {e}")
            return None

    def generate_deployment_report(self) -> dict[str, Any]:
        """Generate comprehensive deployment report."""
        report = {
            "deployment_timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "configuration": {
                "environment": self.config.environment.value,
                "debug": self.config.debug,
                "logging_level": self.config.logging.level,
                "ml_config": self.config.ml.to_dict(),
                "security_config": self.config.security.to_dict(),
            },
            "system_info": {},
            "health_status": {},
            "performance_metrics": {},
            "recommendations": [],
        }

        try:
            # System information
            import psutil

            report["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "disk_total_gb": psutil.disk_usage("/").total / 1024 / 1024 / 1024,
                "python_version": sys.version,
            }

            # Health status
            health_checker = get_health_checker()
            health_status = health_checker.check_system_health()
            report["health_status"] = health_status.to_dict()

            # Recommendations
            recommendations = []
            if health_status.memory_usage_percent > 80:
                recommendations.append("Consider increasing system memory")
            if health_status.disk_usage_percent > 80:
                recommendations.append("Monitor disk usage - consider cleanup")
            if health_status.cpu_usage_percent > 70:
                recommendations.append("High CPU usage detected - monitor performance")

            report["recommendations"] = recommendations

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")

        return report


def main() -> None:
    """Main production management CLI."""
    parser = argparse.ArgumentParser(description="LLKJJ ML Production Manager")
    parser.add_argument(
        "command",
        choices=["setup", "test", "deploy", "health", "report"],
        help="Command to execute",
    )
    parser.add_argument(
        "--env",
        default="production",
        choices=["development", "staging", "production"],
        help="Environment",
    )
    parser.add_argument("--port", type=int, default=8080, help="Health service port")
    parser.add_argument(
        "--scenario",
        default="medium",
        choices=["light", "medium", "heavy", "stress"],
        help="Load test scenario",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    )

    # Create production manager
    manager = ProductionManager(args.env)

    print("🚀 LLKJJ ML Production Manager")
    print(f"Environment: {args.env}")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        if args.command == "setup":
            success = manager.setup_environment()
            exit_code = 0 if success else 1

        elif args.command == "test":
            performance_ok = manager.run_performance_tests()
            load_ok = manager.run_load_tests(args.scenario)
            security_ok = manager.run_security_audit()

            success = performance_ok and load_ok and security_ok
            exit_code = 0 if success else 1

        elif args.command == "deploy":
            success = manager.deploy_production()
            exit_code = 0 if success else 1

        elif args.command == "health":
            process = manager.start_health_service(args.port)
            if process:
                print(f"✅ Health service running on port {args.port}")
                print("Press Ctrl+C to stop...")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping health service...")
                    process.terminate()
                    process.wait()
            exit_code = 0

        elif args.command == "report":
            report = manager.generate_deployment_report()
            print("\n📊 Deployment Report:")
            print(f"Environment: {report['environment']}")
            print(f"Health Status: {report['health_status'].get('status', 'unknown')}")

            # Safely access metrics
            health_metrics = report["health_status"].get("metrics", {})
            if health_metrics:
                print(
                    f"Memory Usage: {health_metrics.get('memory_usage_percent', 0):.1f}%"
                )
                print(f"CPU Usage: {health_metrics.get('cpu_usage_percent', 0):.1f}%")

            if report["recommendations"]:
                print("\n💡 Recommendations:")
                for rec in report["recommendations"]:
                    print(f"  • {rec}")

            # Save full report
            report_file = Path(
                f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w", encoding="utf-8") as f:
                import json

                json.dump(report, f, indent=2)
            print(f"\nFull report saved to: {report_file}")
            exit_code = 0

        else:
            print(f"❌ Unknown command: {args.command}")
            exit_code = 1

        print("-" * 50)
        if exit_code == 0:
            print("✅ Operation completed successfully!")
        else:
            print("❌ Operation failed!")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.exception("Unexpected error in production manager")
        sys.exit(1)


if __name__ == "__main__":
    main()
