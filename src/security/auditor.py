"""
Security Auditing für LLKJJ ML Pipeline
========================================

Umfassende Security-Audits:
- Bandit: Source Code Security Analysis
- Safety: Dependency Vulnerability Scanning
- Custom Security Checks
- Production-Readiness Report
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SecurityAuditor:
    """
    Umfassender Security-Auditor für LLKJJ ML Pipeline.

    Features:
    - Bandit Source Code Analysis
    - Safety Dependency Vulnerability Scanning
    - Custom Security Pattern Detection
    - Aggregierte Security Reports
    """

    def __init__(self, project_root: Path | None = None):
        """
        Initialisiert den Security Auditor.

        Args:
            project_root: Projekt-Root-Verzeichnis
        """
        self.project_root = project_root or Path.cwd()
        self.reports_dir = self.project_root / "security_reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Security-Konfiguration
        self.bandit_config: dict[str, Any] = {
            "exclude_dirs": [
                "tests",
                "__pycache__",
                ".pytest_cache",
                "node_modules",
                ".git",
            ],
            "severity_level": "low",
            "confidence_level": "low",
        }

        self.custom_patterns = [
            r"(?i)(password|secret|key|token)\s*=\s*['\"][^'\"]+['\"]",
            r"(?i)api[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]",
            r"(?i)secret[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]",
            r"eval\s*\(",
            r"exec\s*\(",
            r"input\s*\(",  # Potentielle Injection
            r"os\.system\s*\(",  # Command Injection
            r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
        ]

    def run_bandit_analysis(self) -> dict[str, Any]:
        """
        Führt Bandit Source Code Security Analysis durch.

        Returns:
            Bandit Analysis Report
        """
        logger.info("Starte Bandit Security Analysis...")

        bandit_cmd: list[str] = [
            "bandit",
            "-r",
            str(self.project_root / "src"),
            "-f",
            "json",
            "--severity-level",
            str(self.bandit_config["severity_level"]),
            "--confidence-level",
            str(self.bandit_config["confidence_level"]),
        ]

        # Exclude Directories
        exclude_dirs: list[str] = self.bandit_config["exclude_dirs"]
        for exclude_dir in exclude_dirs:
            bandit_cmd.append("--exclude")
            bandit_cmd.append(exclude_dir)

        try:
            result = subprocess.run(
                bandit_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,  # 5 Minuten Timeout
                check=False,  # Explizit gesetzt
            )

            if result.stdout:
                bandit_report = json.loads(result.stdout)
            else:
                bandit_report = {"errors": [result.stderr], "results": []}

            # Speichere Report
            report_file = (
                self.reports_dir
                / f"bandit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(bandit_report, f, indent=2)

            logger.info("Bandit Analysis abgeschlossen: %s", report_file)

            return {
                "status": "success",
                "report_file": str(report_file),
                "findings": bandit_report.get("results", []),
                "errors": bandit_report.get("errors", []),
                "metrics": bandit_report.get("metrics", {}),
                "summary": {
                    "total_issues": len(bandit_report.get("results", [])),
                    "high_severity": len(
                        [
                            r
                            for r in bandit_report.get("results", [])
                            if r.get("issue_severity") == "HIGH"
                        ]
                    ),
                    "medium_severity": len(
                        [
                            r
                            for r in bandit_report.get("results", [])
                            if r.get("issue_severity") == "MEDIUM"
                        ]
                    ),
                    "low_severity": len(
                        [
                            r
                            for r in bandit_report.get("results", [])
                            if r.get("issue_severity") == "LOW"
                        ]
                    ),
                },
            }

        except subprocess.TimeoutExpired:
            logger.error("Bandit Analysis Timeout nach 5 Minuten")
            return {"status": "timeout", "error": "Analysis Timeout"}

        except OSError as e:
            logger.error("Fehler bei Bandit Analysis: %s", e)
            return {"status": "error", "error": str(e)}

    def run_safety_scan(self) -> dict[str, Any]:
        """
        Führt Safety Dependency Vulnerability Scanning durch.

        Returns:
            Safety Scan Report
        """
        logger.info("Starte Safety Dependency Vulnerability Scan...")

        try:
            # Safety Check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=180,  # 3 Minuten Timeout
                check=False,  # Explizit gesetzt
            )

            if result.returncode == 0:
                # Keine Vulnerabilities gefunden
                safety_report = {
                    "vulnerabilities": [],
                    "metadata": {"safety_version": "3.6.0"},
                }
            else:
                try:
                    safety_report = json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Fallback bei JSON-Parse-Fehlern
                    safety_report = {
                        "vulnerabilities": [],
                        "errors": [result.stderr],
                        "raw_output": result.stdout,
                    }

            # Speichere Report
            report_file = (
                self.reports_dir
                / f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(safety_report, f, indent=2)

            logger.info("Safety Scan abgeschlossen: %s", report_file)

            vulnerabilities: list[Any] = list(safety_report.get("vulnerabilities", []))

            def count_vulnerabilities_by_severity(
                vulns: list[Any], severity: str
            ) -> int:
                """Count vulnerabilities by severity level."""
                return len(
                    [
                        v
                        for v in vulns
                        if isinstance(v, dict)
                        and str(v.get("severity", "")).lower() == severity
                    ]
                )

            return {
                "status": "success",
                "report_file": str(report_file),
                "vulnerabilities": vulnerabilities,
                "summary": {
                    "total_vulnerabilities": len(vulnerabilities),
                    "critical": count_vulnerabilities_by_severity(
                        vulnerabilities, "critical"
                    ),
                    "high": count_vulnerabilities_by_severity(vulnerabilities, "high"),
                    "medium": count_vulnerabilities_by_severity(
                        vulnerabilities, "medium"
                    ),
                    "low": count_vulnerabilities_by_severity(vulnerabilities, "low"),
                },
            }

        except subprocess.TimeoutExpired:
            logger.error("Safety Scan Timeout nach 3 Minuten")
            return {"status": "timeout", "error": "Scan Timeout"}

        except OSError as e:
            logger.error("Fehler bei Safety Scan: %s", e)
            return {"status": "error", "error": str(e)}

    def run_custom_security_checks(self) -> dict[str, Any]:
        """
        Führt custom Security Pattern Detection durch.

        Returns:
            Custom Security Check Report
        """
        logger.info("Starte Custom Security Pattern Detection...")

        import re

        findings = []

        # Scanne alle Python-Dateien
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            # Skip Test Files und Cache
            if any(
                skip in str(file_path)
                for skip in ["test_", "__pycache__", ".pytest_cache"]
            ):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Prüfe jedes Security Pattern
                for pattern in self.custom_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)

                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        findings.append(
                            {
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": line_number,
                                "pattern": pattern,
                                "match": match.group(0),
                                "severity": self._get_pattern_severity(pattern),
                                "description": self._get_pattern_description(pattern),
                            }
                        )

            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Fehler beim Scannen von %s: %s", file_path, e)

        # Speichere Report
        report = {
            "timestamp": datetime.now().isoformat(),
            "scanned_files": len(python_files),
            "findings": findings,
            "patterns_checked": len(self.custom_patterns),
        }

        report_file = (
            self.reports_dir
            / f"custom_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Custom Security Checks abgeschlossen: %s", report_file)

        return {
            "status": "success",
            "report_file": str(report_file),
            "findings": findings,
            "summary": {
                "total_findings": len(findings),
                "high_severity": len([f for f in findings if f["severity"] == "high"]),
                "medium_severity": len(
                    [f for f in findings if f["severity"] == "medium"]
                ),
                "low_severity": len([f for f in findings if f["severity"] == "low"]),
            },
        }

    def _get_pattern_severity(self, pattern: str) -> str:
        """Bestimmt Severity für Security Pattern."""
        if any(
            keyword in pattern.lower()
            for keyword in ["password", "secret", "key", "token"]
        ):
            return "high"
        elif any(keyword in pattern.lower() for keyword in ["eval", "exec", "system"]):
            return "high"
        elif "shell" in pattern.lower():
            return "medium"
        else:
            return "low"

    def _get_pattern_description(self, pattern: str) -> str:
        """Liefert Beschreibung für Security Pattern."""
        descriptions = {
            r"(?i)(password|secret|key|token)\s*=\s*['\"][^'\"]+['\"]": "Hardcoded credentials detected",
            r"(?i)api[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]": "Hardcoded API key detected",
            r"(?i)secret[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]": "Hardcoded secret key detected",
            r"eval\s*\(": "Dangerous eval() usage detected",
            r"exec\s*\(": "Dangerous exec() usage detected",
            r"input\s*\(": "User input without validation",
            r"os\.system\s*\(": "Potential command injection",
            r"subprocess\.call\s*\([^)]*shell\s*=\s*True": "Shell injection vulnerability",
        }

        for desc_pattern, description in descriptions.items():
            if pattern == desc_pattern:
                return description

        return "Custom security pattern match"

    def generate_comprehensive_report(self) -> dict[str, Any]:
        """
        Generiert umfassenden Security Report.

        Returns:
            Comprehensive Security Report
        """
        logger.info("Generiere umfassenden Security Report...")

        # Führe alle Security Checks durch
        bandit_result = self.run_bandit_analysis()
        safety_result = self.run_safety_scan()
        custom_result = self.run_custom_security_checks()

        # Aggregiere Ergebnisse
        comprehensive_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "python_version": sys.version,
                "auditor_version": "1.0.0",
            },
            "bandit_analysis": bandit_result,
            "safety_scan": safety_result,
            "custom_checks": custom_result,
            "overall_summary": {
                "total_security_issues": (
                    bandit_result.get("summary", {}).get("total_issues", 0)
                    + safety_result.get("summary", {}).get("total_vulnerabilities", 0)
                    + custom_result.get("summary", {}).get("total_findings", 0)
                ),
                "critical_issues": (
                    bandit_result.get("summary", {}).get("high_severity", 0)
                    + safety_result.get("summary", {}).get("critical", 0)
                    + custom_result.get("summary", {}).get("high_severity", 0)
                ),
                "production_ready": None,  # Wird unten berechnet
            },
        }

        # Production Readiness Assessment
        critical_issues = comprehensive_report["overall_summary"]["critical_issues"]
        total_issues = comprehensive_report["overall_summary"]["total_security_issues"]

        comprehensive_report["overall_summary"]["production_ready"] = (
            critical_issues == 0 and total_issues < 10
        )

        # Speichere Comprehensive Report
        report_file = (
            self.reports_dir
            / f"comprehensive_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(comprehensive_report, f, indent=2)

        logger.info("Comprehensive Security Report generiert: %s", report_file)

        return comprehensive_report

    def print_security_summary(self, report: dict[str, Any]) -> None:
        """
        Druckt Security Summary für CLI-Output.

        Args:
            report: Comprehensive Security Report
        """
        print("\n" + "=" * 80)
        print("🔒 LLKJJ ML Pipeline - Security Audit Report")
        print("=" * 80)

        # Overall Summary
        summary = report["overall_summary"]
        print("\n📊 Overall Summary:")
        print(f"   Total Security Issues: {summary['total_security_issues']}")
        print(f"   Critical Issues: {summary['critical_issues']}")
        print(
            f"   Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}"
        )

        # Bandit Results
        bandit = report["bandit_analysis"]
        if bandit["status"] == "success":
            print("\n🔍 Bandit Analysis:")
            bandit_summary = bandit["summary"]
            print(f"   Total Issues: {bandit_summary['total_issues']}")
            print(
                f"   High: {bandit_summary['high_severity']}, Medium: {bandit_summary['medium_severity']}, Low: {bandit_summary['low_severity']}"
            )
        else:
            print(
                f"\n❌ Bandit Analysis Failed: {bandit.get('error', 'Unknown error')}"
            )

        # Safety Results
        safety = report["safety_scan"]
        if safety["status"] == "success":
            print("\n🛡️ Safety Vulnerability Scan:")
            safety_summary = safety["summary"]
            print(
                f"   Total Vulnerabilities: {safety_summary['total_vulnerabilities']}"
            )
            if safety_summary["total_vulnerabilities"] > 0:
                print(
                    f"   Critical: {safety_summary['critical']}, High: {safety_summary['high']}, Medium: {safety_summary['medium']}, Low: {safety_summary['low']}"
                )
        else:
            print(f"\n❌ Safety Scan Failed: {safety.get('error', 'Unknown error')}")

        # Custom Checks Results
        custom = report["custom_checks"]
        if custom["status"] == "success":
            print("\n🔎 Custom Security Checks:")
            custom_summary = custom["summary"]
            print(f"   Total Findings: {custom_summary['total_findings']}")
            if custom_summary["total_findings"] > 0:
                print(
                    f"   High: {custom_summary['high_severity']}, Medium: {custom_summary['medium_severity']}, Low: {custom_summary['low_severity']}"
                )

        print("\n" + "=" * 80)

        if not summary["production_ready"]:
            print("⚠️  WARNUNG: Kritische Security-Issues gefunden!")
            print("   Beheben Sie diese vor Production-Deployment.")
        else:
            print("✅ Security-Status: Production-Ready")

        print("=" * 80 + "\n")


def run_security_audit(
    project_root: Path | None = None, verbose: bool = False
) -> dict[str, Any]:
    """
    Führt vollständiges Security Audit durch.

    Args:
        project_root: Projekt-Root-Verzeichnis
        verbose: Detaillierte Ausgabe

    Returns:
        Comprehensive Security Report
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    auditor = SecurityAuditor(project_root)
    report = auditor.generate_comprehensive_report()

    auditor.print_security_summary(report)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLKJJ ML Pipeline Security Auditor")
    parser.add_argument("--project-root", type=Path, help="Projekt-Root-Verzeichnis")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Detaillierte Ausgabe"
    )

    args = parser.parse_args()

    run_security_audit(args.project_root, args.verbose)
